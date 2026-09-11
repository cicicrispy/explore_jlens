"""Interventions: swap and its controls, plus the trace-orchestrating `apply`.

Swap math -- CLAMPED to the clean run (departure from the spec's reference loop, 2026-09-11; see
methods.md): at each band layer, with V = [v_s, v_t] (d x 2) from that layer's lens vectors,
    c_clean = pinv(V) @ h_clean                      the UNEDITED run's coordinates there
    target  = c_clean + alpha * (flip(c_clean) - c_clean)
    h_new   = h + V @ (target - pinv(V) @ h)         the stream's two coordinates are SET to target
whatever the earlier band layers already did. The spec's loop flipped the stream's CURRENT
coordinates at every layer, so over a band each layer undid the one before: at alpha = 1 an even
number of layers comes back near clean, and alpha = 2 multiplies the gap between the two coordinates
by -3 per layer. M1 run validate_20260911-201047 (32 layers) showed both: at alpha = 1 the answer
barely moved, at alpha = 2 ' big' took over. With h_clean = h (a single layer, or the first band
layer) the two are identical. The orthogonal complement of span(V) is untouched by construction.

Logged sizes are those of the clamp measured on the clean run: delta_c_norm = ||target - c_clean||,
delta_h_norm = ||V (target - c_clean)||. The controls are clamps too and are matched to these sizes
(big_nonlabel: its own alpha per position; random_direction: its coordinate along u is set to the
clean value + the size). What was actually written at each layer -- the full edit at the first band
layer, then only a correction for what the model rewrote -- is `changes` in `apply`.

NOTE on `apply`'s kw contract (flagged assumption): lens vectors are layer-dependent (J[l] differs
per layer), so `apply` cannot be handed one fixed v_s/v_t across the whole layer band -- it must
recompute lens_vectors(...) per layer inside the trace, per the spec's reference pseudocode. That
means `apply`'s kwargs are token identifiers (strings or ids), not precomputed vectors; see the
per-kind kw list in `apply`'s docstring. On the real model (M1 run validate_20260911-201047) the
in-trace write reached the model's output.
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
    # The stream's own coordinates at this layer just before the clamp (after the earlier band
    # layers' edits); c_before is the CLEAN run's. Equal at the first band layer.
    c_stream: tuple[float, float] = (0.0, 0.0)


def _basis(v_s: torch.Tensor, v_t: torch.Tensor):
    V = torch.stack([v_s, v_t], dim=-1)  # [d, 2]
    V_pinv = torch.linalg.pinv(V)        # [2, d]
    return V, V_pinv


def _coords(h: torch.Tensor, V_pinv: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bpd,kd->bpk", h, V_pinv)    # c = h @ V_pinv.T, [batch,pos,2]


def _scale(x: torch.Tensor, alpha) -> torch.Tensor:
    if torch.is_tensor(alpha) and alpha.dim() > 0:
        return x * alpha[None, :, None]
    return x * alpha


def _clamp(h, h_clean, V, V_pinv, alpha):
    """(delta, c_clean, target, c_stream): delta = V (target - c_stream) sets h's coordinates in
    span(V) to target = c_clean + alpha * (flip(c_clean) - c_clean)."""
    c_clean = _coords(h_clean, V_pinv)
    target = c_clean + _scale(c_clean[..., [1, 0]] - c_clean, alpha)
    c_stream = _coords(h, V_pinv)
    delta = torch.einsum("bpk,dk->bpd", target - c_stream, V)   # [batch,pos,d]
    return delta, c_clean, target, c_stream


def _apply_mask(delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return delta * mask.to(device=delta.device, dtype=delta.dtype)[None, :, None]


def _alpha_at(alpha, p: int) -> float:
    if torch.is_tensor(alpha) and alpha.dim() > 0:
        return float(alpha[p])
    return float(alpha)


def _make_logs(c_clean, target, c_stream, V, mask, alpha, layer, kind) -> list[InterventionLog]:
    """Per planned position: the clamp's sizes measured on the clean run (see the module docstring)."""
    diff = target - c_clean                               # [batch,pos,2]
    dh = torch.einsum("bpk,dk->bpd", diff, V)[0].norm(dim=-1)  # ||V (target - c_clean)||, [pos]
    logs = []
    for p in mask.nonzero(as_tuple=True)[0].tolist():
        logs.append(
            InterventionLog(
                pos=p,
                layer=layer,
                c_before=tuple(c_clean[0, p].tolist()),
                c_after=tuple(target[0, p].tolist()),
                delta_c_norm=float(diff[0, p].norm()),
                delta_h_norm=float(dh[p]),
                alpha=_alpha_at(alpha, p),
                kind=kind,
                c_stream=tuple(c_stream[0, p].tolist()),
            )
        )
    return logs


def swap(h, mask, v_s, v_t, alpha=1.0, layer=None, h_clean=None):
    """Clamp the (v_s, v_t) coordinates to the clean run's, swapped (module docstring). `h_clean`:
    the unedited stream at this layer; None means `h` itself (a single edited layer)."""
    if torch.equal(v_s, v_t):
        # Swapping a token with itself is a no-op by definition. pinv([v, v]) isn't guaranteed to
        # give bitwise-identical rows in floating point, so short-circuit rather than add ~1e-8 noise.
        logs = identity(h, mask, layer=layer)[1]
        for lg in logs:
            lg.kind, lg.alpha = "swap", float(alpha)
        return h, logs
    h_clean = h if h_clean is None else h_clean
    V, V_pinv = _basis(v_s, v_t)
    delta, c_clean, target, c_stream = _clamp(h, h_clean, V, V_pinv, alpha)
    h_new = h + _apply_mask(delta, mask)
    logs = _make_logs(c_clean, target, c_stream, V, mask, alpha, layer, "swap")
    return h_new, logs


def label_to_present(h, mask, v_source, v_target, alpha=1.0, layer=None, h_clean=None):
    """Same math as swap; v_source is the label being removed, v_target a present non-label token
    (picked by the controls run, read by the human)."""
    h_clean = h if h_clean is None else h_clean
    V, V_pinv = _basis(v_source, v_target)
    delta, c_clean, target, c_stream = _clamp(h, h_clean, V, V_pinv, alpha)
    h_new = h + _apply_mask(delta, mask)
    logs = _make_logs(c_clean, target, c_stream, V, mask, alpha, layer, "label_to_present")
    return h_new, logs


def big_nonlabel(h, mask, v_a, v_b, target_norms, norm_scale=1.0, layer=None, h_clean=None):
    """Swap between two strongly-present non-label tokens, clamped like `swap`, with alpha per
    position chosen so that the clamp's size on the clean run, ||V (target - c_clean)||, is
    norm_scale * target_norms[pos]. Leaves any direction orthogonal to span(v_a, v_b) -- including
    the label coordinate, by construction of v_a/v_b being non-label -- untouched."""
    h_clean = h if h_clean is None else h_clean
    V, V_pinv = _basis(v_a, v_b)
    c_clean = _coords(h_clean, V_pinv)
    base_norm = torch.einsum("bpk,dk->bpd", c_clean[..., [1, 0]] - c_clean, V)[0].norm(dim=-1)  # alpha=1 size, [pos]
    target_norms = target_norms.to(device=h.device, dtype=h.dtype)
    alpha = torch.where(
        base_norm > 0,
        norm_scale * target_norms / base_norm,
        torch.zeros_like(base_norm),
    )
    delta, c_clean, target, c_stream = _clamp(h, h_clean, V, V_pinv, alpha)
    h_new = h + _apply_mask(delta, mask)
    logs = _make_logs(c_clean, target, c_stream, V, mask, alpha, layer, "big_nonlabel")
    return h_new, logs


def random_direction(h, mask, target_norms, seed, layer=None, h_clean=None):
    """Clamp the stream's coordinate along a random unit vector u (fixed per seed+layer) to the
    clean run's value + target_norms[pos] -- on the clean run, adding target_norms[pos] * u. No
    V/label coordinate is defined for this control, so c_before/c_after/c_stream are logged as
    (0.0, 0.0); delta_h_norm is the size on the clean run."""
    h_clean = h if h_clean is None else h_clean
    d = h.shape[-1]
    g = torch.Generator().manual_seed(seed + (layer or 0))
    u = torch.randn(d, generator=g).to(device=h.device, dtype=h.dtype)
    u = u / u.norm()
    target_norms = target_norms.to(device=h.device, dtype=h.dtype)
    target = h_clean @ u + target_norms[None, :]           # [batch,pos]
    delta = (target - h @ u)[..., None] * u[None, None, :]
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
                delta_h_norm=abs(float(target_norms[p])),
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


def clean_states(model, prompt, layers) -> dict:
    """The UNEDITED stream at each of `layers`: {layer: Tensor[1, pos, d] float32}, from one plain
    trace -- where the clamps' targets come from."""
    from . import model as model_mod

    # A plain loop, not a dict comprehension: inside an nnsight trace a comprehension's results
    # never get assigned (see m1_checks.check4_positions).
    saved = {}
    with model.trace(prompt.input_ids):
        for l in sorted(layers):
            saved[l] = model_mod.layer_output(model, l).float().save()
    return saved


def apply(model, lens, prompt, kind: str, layers: list[int], mask, return_changes: bool = False,
          clean: dict | None = None, record: list[int] | None = None, **kw):
    """Register `kind` on layer_output(model, l) for each l in `layers`, ascending, inside one
    trace. Each layer sees the already-edited stream and CLAMPS its coordinates to targets from the
    clean run (module docstring): `clean` = clean_states(model, prompt, layers), or None to record
    them here with one extra plain trace (callers running many cells on one prompt pass them in).
    Returns (logits_at_metric: Tensor[vocab], logs: list[InterventionLog]), plus -- if
    `return_changes` -- `changes`: {layer: [float per position]}, the size ||h_new - h|| of what was
    actually written at EVERY position (not only the planned ones; see `edit_problems`). The logs
    only describe the planned positions, so they cannot show an edit that landed elsewhere; `changes`
    can. Plus -- if `record` (a list of layers, edited or not) -- `states`: {layer: Tensor[pos, d]
    float32}, the stream as the next layer receives it (after that layer's edit, if any).

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

    if kind != "identity" and clean is None:
        clean = clean_states(model, prompt, layers)
    missing = [] if kind == "identity" else [l for l in layers if l not in clean]
    assert not missing, f"no clean state for layers {missing}: the clamps' targets come from the clean run"

    changes: dict[int, list[float]] = {}
    record = sorted(set(record or []))
    states: dict[int, torch.Tensor] = {}
    with model.trace(prompt.input_ids):
        # Ascending over edited and recorded layers together: inside a trace, layers must be reached
        # in the order the model runs them.
        for l in sorted(set(layers) | set(record)):
            env = model_mod.layer_output(model, l)
            if l not in layers:  # recorded only
                states[l] = env.float()[0].save()
                continue
            h = env.float()
            assert h.shape[1] == len(mask), (
                f"the sequence in the trace has {h.shape[1]} positions but the mask has {len(mask)} "
                "-- edits would land on the wrong tokens"
            )
            h_clean = None if kind == "identity" else clean[l]

            if kind == "swap":
                # `pairs=[(s, t), ...]` applies several 2-D swaps in sequence at each layer (e.g. the
                # M1 positive control's big->long and bigger->longer); default is the single pair.
                # Each pair's targets come from the clean run.
                h_new, logs = h, []
                for s_tok, t_tok in kw.get("pairs") or [(kw["s_token"], kw["t_token"])]:
                    v_s = lens_mod.lens_vectors(model, lens, [s_tok], l)[0]
                    v_t = lens_mod.lens_vectors(model, lens, [t_tok], l)[0]
                    h_new, lg = swap(h_new, mask, v_s, v_t, alpha=kw.get("alpha", 1.0), layer=l,
                                     h_clean=h_clean)
                    logs += lg
            elif kind == "label_to_present":
                v_source = lens_mod.lens_vectors(model, lens, [kw["source_token"]], l)[0]
                v_target = lens_mod.lens_vectors(model, lens, [kw["target_token"]], l)[0]
                h_new, logs = label_to_present(
                    h, mask, v_source, v_target, alpha=kw.get("alpha", 1.0), layer=l, h_clean=h_clean
                )
            elif kind == "big_nonlabel":
                v_a = lens_mod.lens_vectors(model, lens, [kw["a_token"]], l)[0]
                v_b = lens_mod.lens_vectors(model, lens, [kw["b_token"]], l)[0]
                tn = kw["target_norms"][l]
                h_new, logs = big_nonlabel(
                    h, mask, v_a, v_b, tn, norm_scale=kw.get("norm_scale", 1.0), layer=l, h_clean=h_clean
                )
            elif kind == "random_direction":
                tn = kw["target_norms"][l]
                h_new, logs = random_direction(h, mask, tn, seed=kw["seed"], layer=l, h_clean=h_clean)
            else:  # identity
                h_new, logs = identity(h, mask, layer=l)

            if return_changes:
                changes[l] = (h_new - h)[0].norm(dim=-1).tolist()
            written = h_new.to(env.dtype)
            model_mod.layer_output(model, l)[:] = written
            if l in record:
                states[l] = written.float()[0].save()
            all_logs.extend(logs)

        logits = model.output.logits[0, prompt.metric_pos].float().save()

    out = (logits, all_logs) + ((changes,) if return_changes else ()) + ((states,) if record else ())
    return out


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
