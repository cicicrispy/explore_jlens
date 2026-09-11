"""M1 lens-validation checks (spec M1 steps 1-4), used by scripts/m1_validate.py.

Each `check*` function runs the model and returns plain rows for parquet. They SAVE GENEROUSLY
(top-`save_k` token lists, the rank of the model's real next token) and never bake in the k a
check is judged at. The `summarize_*` / `band_signatures` / `motor_onset` functions apply those k's
afterwards, from the saved rows only -- so judging a check at a different k never needs the model.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from . import cka as cka_mod
from . import interventions as iv
from . import lens as lens_mod
from . import model as model_mod


def single_id(tokenizer, text: str) -> int:
    """The id of `text` if it is exactly one token, else ValueError."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{text!r} is {len(ids)} tokens under this tokenizer, expected 1: {ids}")
    return ids[0]


def first_single_token(tokenizer, variants: list[str]) -> str | None:
    """The first of `variants` that is a single token (spec: "verify single-token; fall back to
    whichever variants are"), or None."""
    return next((v for v in variants if len(tokenizer.encode(v, add_special_tokens=False)) == 1), None)


# ------------------------------------------------------------------ check 1: final-layer agreement


def check1_final_layer(model, lens, prompts, n_positions: int, seed: int, save_k: int) -> list[dict]:
    """Lens readout at the FINAL layer (J = I) vs the model's ACTUAL output logits, at `n_positions`
    random positions spread evenly over `prompts`. Catches a wrong/missing final norm or a broken
    unembed; comparing against unembed() itself would pass by construction. One row per position
    with both top-`save_k` lists and the largest absolute logit difference."""
    final = model_mod.n_layers(model) - 1
    g = torch.Generator().manual_seed(seed)
    per_prompt = max(1, n_positions // len(prompts))
    rows = []
    for p in prompts:
        pos = torch.randperm(len(p.input_ids), generator=g)[:per_prompt].tolist()
        with model.trace(p.input_ids):
            h = model_mod.layer_output(model, final).float()[0, pos].save()
            real = model.output.logits[0, pos].float().save()
        lens_logits = lens_mod._transport_unembed(model, lens, h, final).float()
        real = real.to(lens_logits.device)
        lv, li = torch.topk(lens_logits, save_k, dim=-1)
        rv, ri = torch.topk(real, save_k, dim=-1)
        diff = (lens_logits - real).abs().max(dim=-1).values
        for i, q in enumerate(pos):
            rows.append({
                "stimulus_id": p.stimulus_id, "question_key": p.question_key, "pos": int(q),
                "lens_topk_ids": li[i].tolist(), "lens_topk_logits": lv[i].tolist(),
                "real_topk_ids": ri[i].tolist(), "real_topk_logits": rv[i].tolist(),
                "max_abs_logit_diff": float(diff[i]),
            })
    return rows


def summarize_check1(df: pd.DataFrame, k: int) -> dict:
    """Top-k overlap per position (0..k) and the largest logit difference, from check1 rows."""
    overlaps = [len(set(list(a)[:k]) & set(list(b)[:k]))
                for a, b in zip(df["lens_topk_ids"], df["real_topk_ids"])]
    return {"k": k, "n_positions": len(overlaps), "overlaps": overlaps,
            "min": int(min(overlaps)), "median": float(np.median(overlaps)), "max": int(max(overlaps)),
            "max_abs_logit_diff": float(df["max_abs_logit_diff"].max())}


# ------------------------------------------------------------------ check 2: readout reproduction


def check2_readout(model, lens, prompt, n_layers: int, save_k: int) -> list[dict]:
    """Top-`save_k` lens readout at every MATRIX position of `prompt`, at `n_layers` evenly spaced
    lens layers. One row per (layer, position)."""
    tok = model.tokenizer
    matrix_positions = [i for i, c in enumerate(prompt.classes) if c == "matrix"]
    idx = np.linspace(0, len(lens.layers) - 1, n_layers).astype(int)
    sampled = [lens.layers[i] for i in idx]
    saved = {}
    with model.trace(prompt.input_ids):
        for l in sampled:
            saved[l] = model_mod.layer_output(model, l).float()[0, matrix_positions].save()
    rows = []
    for l in sampled:
        ids, vals = lens_mod.readout(model, lens, saved[l], l, k=save_k)
        for i, pos in enumerate(matrix_positions):
            top = [int(t) for t in ids[i].tolist()]
            rows.append({
                "stimulus_id": prompt.stimulus_id, "question_key": prompt.question_key,
                "pos": pos, "layer": int(l), "topk_ids": top,
                "topk_text": [tok.decode([t]) for t in top], "topk_logits": vals[i].tolist(),
            })
    return rows


def summarize_check2(df: pd.DataFrame, lang_ids: set, k: int) -> dict:
    """Per layer: fraction of matrix positions with any of `lang_ids` in the readout's top-k."""
    frac = {}
    for layer, sub in df.groupby("layer"):
        frac[int(layer)] = float(np.mean([bool(set(list(t)[:k]) & lang_ids) for t in sub["topk_ids"]]))
    return {"k": k, "fraction_by_layer": frac}


# ------------------------------------------------------------------ check 3: causal positive control


def check3_positive_control(model, lens, cfg: dict, save_k: int) -> dict:
    """The paper's Chinese antonym control, exactly as in the paper: the raw prompt (no chat
    template), the swap at EVERY token position (a deliberate departure from invariant 3's
    skip-the-first-positions rule, for this control only), across the lens layers within
    cfg["layer_range"] of depth, each layer's coordinates clamped to the clean run's, swapped (the
    interventions module docstring). The next alpha in cfg["alphas"] runs only if the previous one did
    not make cfg["expect_swapped"] the top-1 token (spec: "If it fails at α=1, run α=2").

    Where the stream actually changed is measured at every position (apply(return_changes=True));
    any change outside the planned positions raises. Returns rows for: `results` (one per
    condition), `masks` (M0-style per-token rows, `edited` = actually changed at any layer),
    `changes` (per condition, layer, position), `readout` (the lens's top-`save_k` at every covered
    layer and position, per condition: what the lens reads on this prompt, clean and under each
    swap), `ranks` (the rank of each of cfg["readout_tokens"] in that readout -- reported only),
    plus `info`."""
    tok = model.tokenizer
    text = cfg["prompt"]
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = list(enc["input_ids"])
    offsets = [(int(s), int(e)) for s, e in enc["offset_mapping"]]
    n = len(input_ids)
    if cfg["positions"] != "all":
        raise ValueError(f"positive control positions {cfg['positions']!r}: only 'all' (the paper's) is implemented")
    mask = torch.ones(n, dtype=torch.bool)

    pairs = []
    for pr in cfg["pairs"]:
        s, t = first_single_token(tok, pr["source"]), first_single_token(tok, pr["target"])
        if s is not None and t is not None:
            pairs.append((s, t))
    if not pairs:
        raise ValueError(f"no pair in {cfg['pairs']} has single-token variants under this tokenizer")

    depth = model_mod.n_layers(model)
    lo, hi = cfg["layer_range"]
    layers = [l for l in lens.layers if int(lo * depth) <= l < int(hi * depth)]
    if not layers:
        raise ValueError(f"no lens layer falls in layer_range {cfg['layer_range']} of depth {depth}")
    clean_id = single_id(tok, cfg["expect_clean"])
    swapped_id = single_id(tok, cfg["expect_swapped"])
    p = SimpleNamespace(input_ids=input_ids, metric_pos=n - 1)

    def result_row(condition, alpha, logits):
        vals, ids = torch.topk(logits.float(), save_k)
        top = [int(i) for i in ids.tolist()]
        return {"condition": condition, "alpha": alpha, "top1_id": top[0],
                "top1_text": tok.decode([top[0]]),
                "top1_is_expected_clean": top[0] == clean_id,
                "top1_is_expected_swapped": top[0] == swapped_id,
                "topk_ids": top, "topk_text": [tok.decode([t]) for t in top], "topk_logits": vals.tolist()}

    # What the lens reads on this prompt, at every covered layer and position, for the clean pass and
    # each swap: the top-`save_k` readout, and the rank of each of cfg["readout_tokens"] (reported,
    # never used by the check itself).
    probes = {t: tok.encode(t, add_special_tokens=False) for t in cfg.get("readout_tokens", [])}
    probe_ids = {t: ids[0] for t, ids in probes.items() if len(ids) == 1}
    readout_rows, rank_rows = [], []

    def read(condition, stream):
        for l in lens.layers:
            logits_l = lens_mod._transport_unembed(model, lens, stream[l], l).float()  # [n, vocab]
            vals, ids = torch.topk(logits_l, save_k, dim=-1)
            for q in range(n):
                top = [int(i) for i in ids[q].tolist()]
                readout_rows.append({"condition": condition, "layer": int(l), "pos": q,
                                     "token_text": text[offsets[q][0]:offsets[q][1]], "topk_ids": top,
                                     "topk_text": [tok.decode([t]) for t in top],
                                     "topk_logits": vals[q].tolist()})
            for t, i in probe_ids.items():
                target = logits_l[:, i:i + 1]
                ranks = (1 + (logits_l > target).sum(dim=-1)).tolist()
                rank_rows.extend({"condition": condition, "layer": int(l), "pos": q, "token": t, "token_id": i,
                                  "rank": int(ranks[q]), "logit": float(target[q, 0])} for q in range(n))

    clean_stream = {}
    with model.trace(input_ids):
        for l in lens.layers:  # a plain loop: see check4_positions
            clean_stream[l] = model_mod.layer_output(model, l).float()[0].save()
        clean = model.output.logits[0, n - 1].float().save()
    results = [result_row("clean", None, clean)]
    read("clean", clean_stream)
    states = iv.clean_states(model, p, layers)  # the clamps' targets (interventions module docstring)
    masks, change_rows, unchanged_all = [], [], []
    for alpha in cfg["alphas"]:
        condition = f"swap_alpha{alpha:g}"
        logits, _logs, changes, stream = iv.apply(model, lens, p, "swap", layers, mask, return_changes=True,
                                                  clean=states, record=list(lens.layers), pairs=pairs,
                                                  alpha=alpha)
        read(condition, stream)
        outside, unchanged = iv.edit_problems(changes, mask, "swap")
        if outside:
            raise RuntimeError(f"{condition}: the stream changed at unplanned positions "
                               f"(layer, pos, size): {outside[:10]}")
        unchanged_all += [[condition, int(l), int(q)] for l, q in unchanged]
        results.append(result_row(condition, alpha, logits))
        for l, sizes in changes.items():
            change_rows += [{"condition": condition, "layer": int(l), "pos": q, "change_norm": float(sz)}
                            for q, sz in enumerate(sizes)]
        changed = [any(changes[l][q] != 0.0 for l in changes) for q in range(n)]
        masks += [{"stimulus_id": "antonym", "question_key": condition, "pos": q,
                   "token_id": input_ids[q], "token_text": text[offsets[q][0]:offsets[q][1]],
                   "class": "prompt", "edited": changed[q], "is_metric_pos": q == n - 1}
                  for q in range(n)]
        if results[-1]["top1_is_expected_swapped"]:
            break  # the next alpha runs only if this one failed

    info = {"prompt": text, "n_tokens": n, "tokens": [text[s:e] for s, e in offsets],
            "token_ids": input_ids, "pairs": [list(pr) for pr in pairs], "layers": [int(l) for l in layers],
            "alphas_run": [r["alpha"] for r in results[1:]],
            "clean_top1_is_expected": results[0]["top1_is_expected_clean"],
            "passed": any(r["top1_is_expected_swapped"] for r in results[1:]),
            "planned_positions_unchanged": unchanged_all,
            "readout_tokens": list(probe_ids),
            "readout_tokens_not_single": [t for t in probes if t not in probe_ids]}
    return {"results": results, "masks": masks, "changes": change_rows, "info": info,
            "readout": readout_rows, "ranks": rank_rows}


def summarize_check3_readout(ranks: pd.DataFrame, readout: pd.DataFrame, pos: int, layers: list[int],
                             band: list[int]) -> dict:
    """From check3's saved rows: {"ranks": {condition: {token: {layer: rank}}} at position `pos` and
    the given `layers`; "top5": {condition: {layer: [text, ...]}} there; "best": {token: (rank,
    layer, pos)} -- each token's best clean rank over every position and the `band` layers}."""
    at = ranks[(ranks["pos"] == pos) & ranks["layer"].isin(layers)]
    table = {c: {t: dict(zip(g2["layer"].astype(int), g2["rank"].astype(int))) for t, g2 in g.groupby("token", sort=False)}
             for c, g in at.groupby("condition", sort=False)}
    ro = readout[(readout["pos"] == pos) & readout["layer"].isin(layers)]
    top5 = {c: {int(r["layer"]): list(r["topk_text"])[:5] for _, r in g.iterrows()}
            for c, g in ro.groupby("condition", sort=False)}
    clean = ranks[(ranks["condition"] == "clean") & ranks["layer"].isin(band)]
    best = {t: (int(r["rank"]), int(r["layer"]), int(r["pos"]))
            for t, g in clean.groupby("token", sort=False) for r in [g.loc[g["rank"].idxmin()]]}
    return {"ranks": table, "top5": top5, "best": best}


# ------------------------------------------------------------------ check 4: band signatures


def cka_rows(model, lens, n_tokens: int, seed: int) -> list[dict]:
    C, layers = cka_mod.cka_matrix(model, lens, n_tokens=n_tokens, seed=seed)
    return [{"layer_a": int(a), "layer_b": int(b), "cka": float(C[i, j])}
            for i, a in enumerate(layers) for j, b in enumerate(layers)]


def check4_positions(model, lens, prompts) -> pd.DataFrame:
    """At every content (non-template) position of every prompt and every lens layer: the 1-based
    rank of the model's ACTUAL top-1 next token in the lens readout, and the readout's excess
    kurtosis (over the vocab, on LOGITS -- softmax probabilities are heavy-tailed at every layer
    and would hide the signature). Agreement at any k is `rank <= k`, computed afterwards."""
    cols = {k: [] for k in ("stimulus_id", "question_key", "pos", "layer", "actual_top1_id",
                            "rank_of_actual", "kurtosis")}
    for p in prompts:
        pos = [i for i, c in enumerate(p.classes) if c != "template"]
        # A plain loop, not a dict comprehension: inside an nnsight trace a comprehension's results
        # never get assigned (UnboundLocalError after the trace).
        saved = {}
        with model.trace(p.input_ids):
            for l in lens.layers:
                saved[l] = model_mod.layer_output(model, l).float()[0, pos].save()
            actual = model.output.logits[0, pos].float().save()
        actual_top1 = torch.argmax(actual, dim=-1)
        for l in lens.layers:
            logits = lens_mod._transport_unembed(model, lens, saved[l], l).float()  # [P, vocab]
            a = actual_top1.to(logits.device)
            target = logits.gather(1, a[:, None])
            rank = 1 + (logits > target).sum(dim=-1)
            z = (logits - logits.mean(-1, keepdim=True)) / logits.std(-1, unbiased=False, keepdim=True)
            kurt = (z ** 4).mean(-1) - 3.0
            cols["stimulus_id"] += [p.stimulus_id] * len(pos)
            cols["question_key"] += [p.question_key] * len(pos)
            cols["pos"] += pos
            cols["layer"] += [int(l)] * len(pos)
            cols["actual_top1_id"] += a.tolist()
            cols["rank_of_actual"] += rank.tolist()
            cols["kurtosis"] += kurt.tolist()
    return pd.DataFrame(cols)


def band_signatures(pos_df: pd.DataFrame, cka_df: pd.DataFrame, agreement_k: int) -> pd.DataFrame:
    """Per layer: cka_onset_score (mean CKA between the layer and every LATER covered layer -- a
    heuristic summary; the heatmap is primary), agreement_top1, agreement_top{k}, mean kurtosis."""
    C = cka_df.pivot(index="layer_a", columns="layer_b", values="cka").sort_index().sort_index(axis=1)
    Cn = C.values
    onset = {int(l): (float(Cn[i, i + 1:].mean()) if i + 1 < len(C) else float("nan"))
             for i, l in enumerate(C.index)}
    g = pos_df.groupby("layer")
    out = pd.DataFrame({
        "agreement_top1": g["rank_of_actual"].apply(lambda r: float((r <= 1).mean())),
        f"agreement_top{agreement_k}": g["rank_of_actual"].apply(lambda r: float((r <= agreement_k).mean())),
        "kurtosis": g["kurtosis"].mean(),
    }).reset_index()
    out.insert(1, "cka_onset_score", out["layer"].map(onset))
    return out


def motor_onset(df: pd.DataFrame, mid_range: tuple[float, float]) -> int | None:
    """Spec: the first layer where top-1 agreement exceeds the midpoint between its median over
    layers `mid_range` (fractions of the covered layers, e.g. 25-60%) and its value at the last
    covered layer. None if no layer does."""
    n = len(df)
    lo, hi = mid_range
    mid = df.iloc[int(lo * n): int(hi * n) + 1]["agreement_top1"]
    threshold = (mid.median() + df["agreement_top1"].iloc[-1]) / 2
    above = df[df["agreement_top1"] > threshold]["layer"]
    return int(above.iloc[0]) if len(above) else None
