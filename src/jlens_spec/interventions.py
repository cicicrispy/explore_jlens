"""Interventions: swap and its controls, plus the trace-orchestrating `apply`.

Swap math (exact, per spec): V=[v_s,v_t] (d x 2), c = pinv(V) @ h, h_new = h + alpha * V @ (flip(c) - c).
The orthogonal complement of span(V) is untouched by construction.

NOTE on `apply`'s kw contract (flagged assumption): lens vectors are layer-dependent (J[l] differs
per layer), so `apply` cannot be handed one fixed v_s/v_t across the whole layer band -- it must
recompute lens_vectors(...) per layer inside the trace, per the spec's reference pseudocode. That
means `apply`'s kwargs are token identifiers (strings or ids), not precomputed vectors; see the
per-kind kw list in `apply`'s docstring. This has not been exercised against a real nnsight trace
yet -- verify on first real run and adjust if nnsight's envoy semantics differ.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class InterventionLog:
    pos: int
    layer: int | None
    c_before: tuple[float, float]
    c_after: tuple[float, float]
    delta_c_norm: float
    delta_h_norm: float
    alpha: float
    kind: str


def _basis(v_s: torch.Tensor, v_t: torch.Tensor):
    V = torch.stack([v_s, v_t], dim=-1)  # [d, 2]
    V_pinv = torch.linalg.pinv(V)        # [2, d]
    return V, V_pinv


def _swap_delta(h: torch.Tensor, V: torch.Tensor, V_pinv: torch.Tensor, alpha):
    c = torch.einsum("bpd,kd->bpk", h, V_pinv)       # c = h @ V_pinv.T, [batch,pos,2]
    c_flip = c[..., [1, 0]]
    diff = c_flip - c
    delta = torch.einsum("bpk,dk->bpd", diff, V)      # (flip(c) - c) @ V.T, [batch,pos,d]
    if torch.is_tensor(alpha) and alpha.dim() > 0:
        delta = delta * alpha[None, :, None]
    else:
        delta = delta * alpha
    return delta, c, c_flip


def _apply_mask(delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return delta * mask.to(device=delta.device, dtype=delta.dtype)[None, :, None]


def _alpha_at(alpha, p: int) -> float:
    if torch.is_tensor(alpha) and alpha.dim() > 0:
        return float(alpha[p])
    return float(alpha)


def _make_logs(c, c_flip, mask, alpha, delta_h, layer, kind) -> list[InterventionLog]:
    logs = []
    for p in mask.nonzero(as_tuple=True)[0].tolist():
        a = _alpha_at(alpha, p)
        c_before = tuple(c[0, p].tolist())
        diff = (c_flip[0, p] - c[0, p]) * a
        c_after = tuple((c[0, p] + diff).tolist())
        logs.append(
            InterventionLog(
                pos=p,
                layer=layer,
                c_before=c_before,
                c_after=c_after,
                delta_c_norm=float(diff.norm()),
                delta_h_norm=float(delta_h[0, p].norm()),
                alpha=a,
                kind=kind,
            )
        )
    return logs


def swap(h, mask, v_s, v_t, alpha=1.0, layer=None):
    if torch.equal(v_s, v_t):
        # Swapping a token with itself is a no-op by definition. pinv([v, v]) isn't guaranteed to
        # give bitwise-identical rows in floating point, so short-circuit rather than add ~1e-8 noise.
        logs = identity(h, mask, layer=layer)[1]
        for lg in logs:
            lg.kind, lg.alpha = "swap", float(alpha)
        return h, logs
    V, V_pinv = _basis(v_s, v_t)
    delta, c, c_flip = _swap_delta(h, V, V_pinv, alpha)
    delta = _apply_mask(delta, mask)
    h_new = h + delta
    logs = _make_logs(c, c_flip, mask, alpha, delta, layer, "swap")
    return h_new, logs


def label_to_present(h, mask, v_source, v_target, alpha=1.0, layer=None):
    """Same math as swap; v_source is the label being removed, v_target a present non-label token
    with loading below the label's (chosen at M2, config-approved by the human)."""
    V, V_pinv = _basis(v_source, v_target)
    delta, c, c_flip = _swap_delta(h, V, V_pinv, alpha)
    delta = _apply_mask(delta, mask)
    h_new = h + delta
    logs = _make_logs(c, c_flip, mask, alpha, delta, layer, "label_to_present")
    return h_new, logs


def big_nonlabel(h, mask, v_a, v_b, target_norms, norm_scale=1.0, layer=None):
    """Swap between two strongly-present non-label tokens, scaled so delta_h_norm ==
    norm_scale * target_norms[pos]. Leaves any direction orthogonal to span(v_a, v_b) -- including
    the label coordinate, by construction of v_a/v_b being non-label -- untouched."""
    V, V_pinv = _basis(v_a, v_b)
    delta_unit, c, c_flip = _swap_delta(h, V, V_pinv, alpha=1.0)
    target_norms = target_norms.to(device=h.device, dtype=h.dtype)
    base_norm = delta_unit[0].norm(dim=-1)  # [pos]
    alpha = torch.where(
        base_norm > 0,
        norm_scale * target_norms / base_norm,
        torch.zeros_like(base_norm),
    )
    delta = delta_unit * alpha[None, :, None]
    delta = _apply_mask(delta, mask)
    h_new = h + delta
    logs = _make_logs(c, c_flip, mask, alpha, delta, layer, "big_nonlabel")
    return h_new, logs


def random_direction(h, mask, target_norms, seed, layer=None):
    """Add a random unit vector (fixed per seed+layer) scaled to target_norms[pos]. No V/label
    coordinate is defined for this control, so c_before/c_after are logged as (0.0, 0.0)."""
    d = h.shape[-1]
    g = torch.Generator().manual_seed(seed + (layer or 0))
    u = torch.randn(d, generator=g).to(device=h.device, dtype=h.dtype)
    u = u / u.norm()
    target_norms = target_norms.to(device=h.device, dtype=h.dtype)
    delta = target_norms[None, :, None] * u[None, None, :]
    delta = _apply_mask(delta, mask)
    h_new = h + delta
    logs = []
    for p in mask.nonzero(as_tuple=True)[0].tolist():
        logs.append(
            InterventionLog(
                pos=p,
                layer=layer,
                c_before=(0.0, 0.0),
                c_after=(0.0, 0.0),
                delta_c_norm=0.0,
                delta_h_norm=float(delta[0, p].norm()),
                alpha=1.0,
                kind="random_direction",
            )
        )
    return h_new, logs


def identity(h, mask, layer=None):
    logs = []
    for p in mask.nonzero(as_tuple=True)[0].tolist():
        logs.append(
            InterventionLog(
                pos=p,
                layer=layer,
                c_before=(0.0, 0.0),
                c_after=(0.0, 0.0),
                delta_c_norm=0.0,
                delta_h_norm=0.0,
                alpha=0.0,
                kind="identity",
            )
        )
    return h, logs


_KIND_FUNCS = {
    "swap": swap,
    "label_to_present": label_to_present,
    "big_nonlabel": big_nonlabel,
    "random_direction": random_direction,
    "identity": identity,
}


def apply(model, lens, prompt, kind: str, layers: list[int], mask, return_changes: bool = False, **kw):
    """Register `kind` on layer_output(model, l) for each l in `layers`, ascending, inside one
    trace (each layer sees the already-edited stream -- clamped). Returns
    (logits_at_metric: Tensor[vocab], logs: list[InterventionLog]), plus -- if `return_changes` --
    `changes`: {layer: [float per position]}, the size ||h_new - h|| of what was actually written at
    EVERY position (not only the planned ones; see `edit_problems`). The logs only describe the
    planned positions, so they cannot show an edit that landed elsewhere; `changes` can.

    Always asserts that the sequence inside the trace is exactly as long as `mask` (i.e. the prompt
    the mask was built on) -- a shifted sequence would put every edit on the wrong token.

    kw by kind:
      swap:             s_token, t_token, alpha=1.0  (or pairs=[(s, t), ...] for several swaps)
      label_to_present: source_token, target_token, alpha=1.0
      big_nonlabel:     a_token, b_token, target_norms: dict[int, Tensor[pos]], norm_scale=1.0
      random_direction: target_norms: dict[int, Tensor[pos]], seed: int
      identity:         (none)
    """
    assert kind in _KIND_FUNCS, f"unknown intervention kind {kind!r}"
    from . import model as model_mod
    from . import lens as lens_mod

    all_logs: list[InterventionLog] = []
    uncovered = [l for l in layers if l not in lens.J]
    assert not uncovered, (
        f"layers {uncovered} are not covered by the lens; edits there would not be J-lens edits, "
        "and the final layer must never be written to"
    )

    changes: dict[int, list[float]] = {}
    with model.trace(prompt.input_ids):
        for l in sorted(layers):
            env = model_mod.layer_output(model, l)
            h = env.float()
            assert h.shape[1] == len(mask), (
                f"the sequence in the trace has {h.shape[1]} positions but the mask has {len(mask)} "
                "-- edits would land on the wrong tokens"
            )

            if kind == "swap":
                # `pairs=[(s, t), ...]` applies several 2-D swaps in sequence at each layer (e.g. the
                # M1 positive control's big->long and bigger->longer); default is the single pair.
                h_new, logs = h, []
                for s_tok, t_tok in kw.get("pairs") or [(kw["s_token"], kw["t_token"])]:
                    v_s = lens_mod.lens_vectors(model, lens, [s_tok], l)[0]
                    v_t = lens_mod.lens_vectors(model, lens, [t_tok], l)[0]
                    h_new, lg = swap(h_new, mask, v_s, v_t, alpha=kw.get("alpha", 1.0), layer=l)
                    logs += lg
            elif kind == "label_to_present":
                v_source = lens_mod.lens_vectors(model, lens, [kw["source_token"]], l)[0]
                v_target = lens_mod.lens_vectors(model, lens, [kw["target_token"]], l)[0]
                h_new, logs = label_to_present(
                    h, mask, v_source, v_target, alpha=kw.get("alpha", 1.0), layer=l
                )
            elif kind == "big_nonlabel":
                v_a = lens_mod.lens_vectors(model, lens, [kw["a_token"]], l)[0]
                v_b = lens_mod.lens_vectors(model, lens, [kw["b_token"]], l)[0]
                tn = kw["target_norms"][l]
                h_new, logs = big_nonlabel(
                    h, mask, v_a, v_b, tn, norm_scale=kw.get("norm_scale", 1.0), layer=l
                )
            elif kind == "random_direction":
                tn = kw["target_norms"][l]
                h_new, logs = random_direction(h, mask, tn, seed=kw["seed"], layer=l)
            else:  # identity
                h_new, logs = identity(h, mask, layer=l)

            if return_changes:
                changes[l] = (h_new - h)[0].norm(dim=-1).tolist()
            model_mod.layer_output(model, l)[:] = h_new.to(env.dtype)
            all_logs.extend(logs)

        logits = model.output.logits[0, prompt.metric_pos].float().save()

    if return_changes:
        return logits, all_logs, changes
    return logits, all_logs


def edit_problems(changes: dict, mask, kind: str) -> tuple[list, list]:
    """Compare where the stream actually changed (`changes` from `apply(..., return_changes=True)`)
    with where it was planned to (`mask`). Returns (outside, unchanged):
      outside   -- [(layer, pos, size)] positions NOT in the mask whose value changed at all. Any
                   entry means the edit spilled onto unplanned tokens: callers must treat this as a
                   hard failure.
      unchanged -- [(layer, pos)] planned positions that did not change. Expected (and not
                   reported) for `identity`, which edits nothing by design; for other kinds it is
                   reported, not failed (e.g. a swap whose two coordinates happen to be equal)."""
    planned = [bool(m) for m in mask]
    outside, unchanged = [], []
    for layer, sizes in sorted(changes.items()):
        for pos, size in enumerate(sizes):
            if not planned[pos] and size != 0.0:  # exact: unplanned positions get delta * 0
                outside.append((layer, pos, size))
            elif planned[pos] and size == 0.0 and kind != "identity":
                unchanged.append((layer, pos))
    return outside, unchanged
