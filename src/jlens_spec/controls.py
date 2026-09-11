"""Automatic control-token selection for M3 -- no human review (a user-approved departure from the
spec's "human approves controls_proposed.yaml" step; see README "Departures from the spec").

Controls are picked at the start of each position set's POSITIVES run (scripts/m3_grid.py), saved in
that run's controls/ folder before any cell runs, and reused by the matching anomaly run.

Each treatment swap exchanges the language being REMOVED (s) with the one SWAPPED IN (t). With the
treatment pair (es word, fr word) there are four treatment rows per position set:
    passage es, m2i: s = es word, t = fr word        passage es, i2m: s = fr word, t = es word
    passage fr, m2i: s = fr word, t = es word        passage fr, i2m: s = es word, t = fr word
(runner._resolve_pair_words is the same mapping.) ONE set of controls serves all four rows, so every
rule below must hold in every row.

Measures -- always within the run's band only, at the positions the treatment edits:
- layer coverage of token x on one prompt = the fraction of the band's layers at which x is among the
  lens readout's top-`pool_k` tokens at >= 1 edited position (from M2's saved top-100 readout). A
  row's value = the mean over the prompts of that row's passage language (8 passages x 4 questions).
- |Δc| of a swap between tokens s and x at one position and layer = ||flip(c) - c|| with
  c = pinv([v_s, v_x]) h -- the same quantity M3 logs as delta_c_norm -- here from a CLEAN forward
  pass (M3 then measures the real, clamped value). Mean over edited positions x band layers per
  prompt, then over the row's prompts.

Candidates: every token in the top-`pool_k` readout at the edited positions within the band, on any
of the 64 prompts, minus `control_ineligible` tokens and any token whose text contains one of them
(case-insensitive). Tokenizer special tokens and tokens whose text does not re-tokenize to exactly
themselves are never used as controls; they are listed separately (`excluded_special`).

label_to_present -- keeps the removed label s; the control token x stands in for the swapped-in t:
    coverage ratio = cov(x) / cov(t),   |Δc| ratio = |Δc|(s, x) / |Δc|(s, t),   in every row.
big_nonlabel -- a control pair (a, b), a standing in for s and b for t. The swap is symmetric and the
    four rows swap the roles of the two language tokens, so each member must reach the bar against
    both: coverage ratio of a member = cov(member) / max(cov(s), cov(t)), in every row. The edit is
    scaled to at least the treatment's ||Δh|| at every position and layer (norm_scale), so only
    coverage has a bar; among pairs in the same tier, the one whose UNSCALED |Δc| is closest to the
    treatment's is preferred (so the scale factor stays near 1). Pairs are formed from the
    `pool_size` members with the widest coverage; the chosen pairs share no token.

Tiers -- selection never stops:
    1 = every ratio >= bars[0] (100%) in every row;  2 = every ratio >= bars[1] (75%);
    3 = the closest remaining, flagged "below 75%".
Within tiers 1-2 the CLOSEST to the treatment is preferred: the smallest mean over the four rows of
|log(|Δc| ratio)|. Tier 3 is ordered by its smallest ratio, largest first (closest to the bar).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch

from . import lens as lens_mod
from . import model as model_mod

ROWS = ("es_m2i", "es_i2m", "fr_m2i", "fr_i2m")
TIER_LABELS = {1: ">=100%", 2: ">=75%", 3: "below 75%"}


def treatment_rows(pair_words) -> list[dict]:
    """The four treatment rows for the pair (es word, fr word): which token each removes and which
    it swaps in."""
    es, fr = pair_words
    return [
        {"row": "es_m2i", "matrix_lang": "es", "direction": "m2i", "removed": es, "swapped_in": fr},
        {"row": "es_i2m", "matrix_lang": "es", "direction": "i2m", "removed": fr, "swapped_in": es},
        {"row": "fr_m2i", "matrix_lang": "fr", "direction": "m2i", "removed": fr, "swapped_in": es},
        {"row": "fr_i2m", "matrix_lang": "fr", "direction": "i2m", "removed": es, "swapped_in": fr},
    ]


def _tier(worst: np.ndarray, bars) -> np.ndarray:
    return np.where(worst >= bars[0], 1, np.where(worst >= bars[1], 2, 3))


# ------------------------------------------------------------------------------- coverage


def coverage(topk, edited_classes, band_layers, skip_first: int, pool_k: int) -> pd.DataFrame:
    """Per prompt and token: in how many of the band's layers the token is among the top-`pool_k`
    readout at >= 1 edited position. `topk` = M2's saved top-k readout (pyarrow Table or DataFrame
    with stimulus_id, question_key, pos, class, layer, topk_ids), any number of prompts.
    Returns columns stimulus_id, question_key, token_id, n_layers (only tokens seen at least once)."""
    import pyarrow as pa

    df = topk.select(["stimulus_id", "question_key", "pos", "class", "layer"]).to_pandas() \
        if isinstance(topk, pa.Table) else topk[["stimulus_id", "question_key", "pos", "class", "layer"]]
    if isinstance(topk, pa.Table):
        col = topk.column("topk_ids").combine_chunks()
        k = col.type.list_size
        ids = col.flatten().to_numpy().reshape(-1, k)
    else:
        ids = np.stack([np.asarray(v) for v in topk["topk_ids"]])
    keep = (df["class"].isin(list(edited_classes)) & df["layer"].isin(list(band_layers))
            & (df["pos"] >= skip_first)).to_numpy()
    df, ids = df[keep].reset_index(drop=True), ids[keep][:, :pool_k].astype(np.int64)
    if len(df) == 0:
        return pd.DataFrame(columns=["stimulus_id", "question_key", "token_id", "n_layers"])
    prompt_codes, prompts = pd.factorize(df["stimulus_id"] + "\t" + df["question_key"])
    layer_codes, _ = pd.factorize(df["layer"])
    vocab_bound = int(ids.max()) + 1
    n_lay = int(layer_codes.max()) + 1
    key = ((prompt_codes[:, None].astype(np.int64) * n_lay + layer_codes[:, None]) * vocab_bound + ids).reshape(-1)
    uniq = np.unique(key)                                     # distinct (prompt, layer, token)
    pt = (uniq // (vocab_bound * n_lay)) * vocab_bound + uniq % vocab_bound
    pt_uniq, n_layers = np.unique(pt, return_counts=True)     # distinct (prompt, token) -> #layers
    p_idx, tok = pt_uniq // vocab_bound, pt_uniq % vocab_bound
    sid_q = np.array([p.split("\t") for p in prompts], dtype=object)
    return pd.DataFrame({"stimulus_id": sid_q[p_idx, 0], "question_key": sid_q[p_idx, 1],
                         "token_id": tok.astype(np.int64), "n_layers": n_layers.astype(np.int64)})


def coverage_by_language(cov: pd.DataFrame, prompts_by_lang: dict, n_band_layers: int) -> pd.DataFrame:
    """Mean coverage fraction per passage language: rows = token_id, columns cov_es / cov_fr. A token
    absent from a prompt counts 0 for that prompt. `prompts_by_lang` = {"es": [(stimulus_id,
    question_key), ...], "fr": [...]}."""
    out = {}
    for lang, keys in prompts_by_lang.items():
        keyset = {f"{s}\t{q}" for s, q in keys}
        sub = cov[(cov["stimulus_id"] + "\t" + cov["question_key"]).isin(keyset)]
        out[f"cov_{lang}"] = sub.groupby("token_id")["n_layers"].sum() / (n_band_layers * len(keys))
    return pd.DataFrame(out).fillna(0.0)


# ----------------------------------------------------------------------------- candidates


def classify_tokens(token_ids, tokenizer, ineligible) -> pd.DataFrame:
    """For each id: its text and why it can't be a control ('' = eligible). Special tokens and tokens
    that don't re-tokenize to themselves are kept apart ('special token' / 'does not re-tokenize to
    itself') -- never controls, listed separately."""
    special = set(tokenizer.all_special_ids) | set(getattr(tokenizer, "added_tokens_decoder", {}) or {})
    names = [n.strip().lower() for n in ineligible if n.strip()]
    ineligible = set(ineligible)
    rows = []
    for tid in token_ids:
        tid = int(tid)
        text = tokenizer.decode([tid])
        if tid in special:
            reason = "special token"
        elif tokenizer.encode(text, add_special_tokens=False) != [tid]:
            reason = "does not re-tokenize to itself"
        elif text in ineligible:
            reason = "control_ineligible"
        elif any(n in text.lower() for n in names):
            reason = "contains a language name"
        else:
            reason = ""
        rows.append({"token_id": tid, "token": text, "excluded_reason": reason})
    return pd.DataFrame(rows, columns=["token_id", "token", "excluded_reason"])


def big_nonlabel_pool(cands: pd.DataFrame, treat: dict, pool_size: int, bars) -> pd.DataFrame:
    """The `pool_size` big_nonlabel members with the best coverage tier, then the widest mean
    coverage. `cands`: eligible candidates with cov_<row> columns; `treat[row]` has cov_removed and
    cov_swapped_in. Adds columns member_worst_cov_ratio and member_tier."""
    ratios = []
    for r in ROWS:
        ref = max(treat[r]["cov_removed"], treat[r]["cov_swapped_in"])
        ratios.append(np.full(len(cands), np.inf) if ref == 0 else cands[f"cov_{r}"].to_numpy() / ref)
    worst = np.min(np.stack(ratios), axis=0) if len(cands) else np.array([])
    out = cands.assign(member_worst_cov_ratio=worst, member_tier=_tier(worst, bars),
                       mean_cov=cands[[f"cov_{r}" for r in ROWS]].mean(axis=1))
    return out.sort_values(["member_tier", "mean_cov"], ascending=[True, False]).head(pool_size)


# ------------------------------------------------------------------------- |Δc| estimates


def _abs_dc(a, b, c, x, y):
    """|Δc| = sqrt(2) |c_s - c_x| for the swap basis V = [v_s, v_x], from dot products: a = v_s.v_s,
    b = v_s.v_x, c = v_x.v_x, x = h.v_s, y = h.v_x (c = pinv(V) h = (V^T V)^-1 V^T h). Broadcasts."""
    D = a * c - b * b
    return math.sqrt(2.0) * torch.abs(((c + b) * x - (a + b) * y) / D)


def estimate_delta_c(model, lens, prompts, masks, band_layers, cand_ids, pair_ids, pool_ids) -> dict:
    """Clean-pass |Δc| estimates, averaged over each prompt's edited positions x band layers.

    prompts/masks: parallel lists (Prompt, bool mask over its positions). cand_ids: label_to_present
    candidates. pair_ids: (es word id, fr word id) of the treatment pair. pool_ids: big_nonlabel pool.
    Returns per prompt (lists parallel to `prompts`):
      ltp_removed_es / ltp_removed_fr: np.array[len(cand_ids)] -- |Δc|(removed label, candidate)
      treat: float -- |Δc|(es word, fr word), the treatment swap
      pairs: list of (i, j) index pairs into pool_ids, and pair_dc: np.array[n_pairs] per prompt.
    A candidate (nearly) parallel to the label gives a division by ~0 -> inf/NaN; callers drop NaN."""
    dev = model_mod.device_of(model)
    # h at the edited positions for every band layer, per prompt, kept on the CPU until used.
    hs = []
    for p, m in zip(prompts, masks):
        pos = [i for i, on in enumerate(m.tolist()) if on]
        saved = {}
        with model.trace(p.input_ids):
            for l in band_layers:
                saved[l] = model_mod.layer_output(model, l).float()[0, pos].save()
        hs.append({l: saved[l].detach().cpu() for l in band_layers})

    n_p, n_c = len(prompts), len(cand_ids)
    pairs = [(i, j) for i in range(len(pool_ids)) for j in range(i + 1, len(pool_ids))]
    acc = {"es": np.zeros((n_p, n_c)), "fr": np.zeros((n_p, n_c)), "treat": np.zeros(n_p),
           "pairs": np.zeros((n_p, len(pairs)))}
    counts = np.zeros(n_p)
    pi = torch.tensor([i for i, _ in pairs], dtype=torch.long)
    pj = torch.tensor([j for _, j in pairs], dtype=torch.long)
    for l in band_layers:
        Vc = lens_mod.lens_vectors(model, lens, list(cand_ids), l).to(dev) if n_c else None  # [T, d]
        ves, vfr = lens_mod.lens_vectors(model, lens, list(pair_ids), l).to(dev)
        Vq = lens_mod.lens_vectors(model, lens, list(pool_ids), l).to(dev) if len(pool_ids) else None
        a_es, a_fr, g = ves @ ves, vfr @ vfr, ves @ vfr
        if Vc is not None:
            cc = (Vc * Vc).sum(-1)
            b_es, b_fr = Vc @ ves, Vc @ vfr
        if Vq is not None and pairs:
            G = Vq @ Vq.T
            pa_, pc_, pb_ = G[pi, pi], G[pj, pj], G[pi, pj]
        for k in range(n_p):
            H = hs[k][l].to(dev)                                   # [P, d]
            if H.shape[0] == 0:
                continue
            x_es, x_fr = H @ ves, H @ vfr                          # [P]
            acc["treat"][k] += float(_abs_dc(a_es, g, a_fr, x_es, x_fr).sum())
            if Vc is not None:
                Y = H @ Vc.T                                       # [P, T]
                acc["es"][k] += _abs_dc(a_es, b_es, cc, x_es[:, None], Y).sum(0).cpu().numpy()
                acc["fr"][k] += _abs_dc(a_fr, b_fr, cc, x_fr[:, None], Y).sum(0).cpu().numpy()
            if Vq is not None and pairs:
                Yq = H @ Vq.T                                      # [P, Q]
                acc["pairs"][k] += _abs_dc(pa_, pb_, pc_, Yq[:, pi], Yq[:, pj]).sum(0).cpu().numpy()
            counts[k] += H.shape[0]
    denom = np.maximum(counts, 1)
    return {"ltp_removed_es": acc["es"] / denom[:, None], "ltp_removed_fr": acc["fr"] / denom[:, None],
            "treat": acc["treat"] / denom, "pairs": pairs, "pair_dc": acc["pairs"] / denom[:, None],
            "n_positions_x_layers": counts}


def by_language(values: np.ndarray, langs: list[str], lang: str) -> np.ndarray:
    """Mean over the prompts whose passage language is `lang` (axis 0)."""
    idx = [i for i, g in enumerate(langs) if g == lang]
    return np.asarray(values)[idx].mean(axis=0)


def with_coverage(cands: pd.DataFrame, cov_lang: pd.DataFrame, pair_words, pair_ids) -> tuple[pd.DataFrame, dict]:
    """Step 1 (no model): per-row coverage. Adds cov_<row> to `cands` (token_id, token, ...) and
    returns treat = {row: {removed, swapped_in, cov_removed, cov_swapped_in, ...}}."""
    word_id = dict(zip(pair_words, pair_ids))

    def cov_of(tid, lang):
        col = f"cov_{lang}"
        return float(cov_lang[col].get(tid, 0.0)) if col in cov_lang else 0.0

    cands = cands.copy()
    treat = {}
    for row in treatment_rows(pair_words):
        r, lang = row["row"], row["matrix_lang"]
        treat[r] = {**row, "cov_removed": cov_of(word_id[row["removed"]], lang),
                    "cov_swapped_in": cov_of(word_id[row["swapped_in"]], lang)}
        cands[f"cov_{r}"] = [cov_of(t, lang) for t in cands["token_id"]]
    return cands, treat


def with_delta_c(cands: pd.DataFrame, treat: dict, est: dict, langs: list[str], pair_words,
                 pool_ids) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Step 2 (after estimate_delta_c): per-row |Δc|. `cands` must be in the order `est` was computed
    for; `langs` = each estimated prompt's passage language. Adds dc_<row> to `cands` and `dc` to
    each treat row; returns the pair table (a_id, b_id, dc_<row>) for big_nonlabel."""
    es_word = pair_words[0]
    cands, treat = cands.copy(), {r: dict(v) for r, v in treat.items()}
    pairs = pd.DataFrame({"a_id": [int(pool_ids[i]) for i, _ in est["pairs"]],
                          "b_id": [int(pool_ids[j]) for _, j in est["pairs"]]}, dtype=np.int64)
    for row in treatment_rows(pair_words):
        r, lang = row["row"], row["matrix_lang"]
        removed_key = "ltp_removed_es" if row["removed"] == es_word else "ltp_removed_fr"
        treat[r]["dc"] = float(by_language(est["treat"], langs, lang))
        cands[f"dc_{r}"] = by_language(est[removed_key], langs, lang) if len(cands) else []
        pairs[f"dc_{r}"] = by_language(est["pair_dc"], langs, lang) if len(pairs) else []
    return cands, treat, pairs


# ------------------------------------------------------------------------------ selection


def select_label_to_present(cands: pd.DataFrame, treat: dict, n: int, bars) -> pd.DataFrame:
    """Rank label_to_present candidates and return the top `n` (see the module docstring).
    `cands` columns: token_id, token, cov_<row>, dc_<row> for each row; `treat[row]` has
    cov_swapped_in and dc. Adds per-row ratios, worst_ratio, tier, tier_label, closeness."""
    c = cands.copy()
    ratios = []
    for r in ROWS:
        ref_cov = treat[r]["cov_swapped_in"]
        c[f"cov_ratio_{r}"] = np.inf if ref_cov == 0 else c[f"cov_{r}"] / ref_cov
        c[f"dc_ratio_{r}"] = c[f"dc_{r}"] / treat[r]["dc"]
        ratios += [c[f"cov_ratio_{r}"], c[f"dc_ratio_{r}"]]
    c = c[np.isfinite(c[[f"dc_ratio_{r}" for r in ROWS]]).all(axis=1)]
    c["worst_ratio"] = c[[f"cov_ratio_{r}" for r in ROWS] + [f"dc_ratio_{r}" for r in ROWS]].min(axis=1)
    c["tier"] = _tier(c["worst_ratio"].to_numpy(), bars)
    c["closeness"] = np.mean([np.abs(np.log(c[f"dc_ratio_{r}"].clip(lower=1e-12))) for r in ROWS], axis=0)
    c["order"] = np.where(c["tier"] < 3, c["closeness"], -c["worst_ratio"])
    c = c.sort_values(["tier", "order"]).head(n).drop(columns="order")
    c["tier_label"] = c["tier"].map(TIER_LABELS)
    return c.reset_index(drop=True)


def select_big_nonlabel(pool: pd.DataFrame, pair_dc: pd.DataFrame, treat: dict, n: int) -> pd.DataFrame:
    """Pick `n` pairs that share no token (see the module docstring). `pool`: output of
    big_nonlabel_pool. `pair_dc`: columns a_id, b_id, dc_<row> (unscaled |Δc| of the pair's swap).
    A pair's tier is the worse of its two members' tiers."""
    tiers = dict(zip(pool["token_id"], pool["member_tier"]))
    worst = dict(zip(pool["token_id"], pool["member_worst_cov_ratio"]))
    text = dict(zip(pool["token_id"], pool["token"]))
    p = pair_dc.copy()
    p["tier"] = [max(tiers[a], tiers[b]) for a, b in zip(p["a_id"], p["b_id"])]
    p["worst_cov_ratio"] = [min(worst[a], worst[b]) for a, b in zip(p["a_id"], p["b_id"])]
    for r in ROWS:
        p[f"dc_ratio_{r}"] = p[f"dc_{r}"] / treat[r]["dc"]
    p = p[np.isfinite(p[[f"dc_ratio_{r}" for r in ROWS]]).all(axis=1)]
    p["closeness"] = np.mean([np.abs(np.log(p[f"dc_ratio_{r}"].clip(lower=1e-12))) for r in ROWS], axis=0)
    p["order"] = np.where(p["tier"] < 3, p["closeness"], -p["worst_cov_ratio"])
    p = p.sort_values(["tier", "order"])
    chosen, used = [], set()
    for row in p.itertuples(index=False):
        if row.a_id in used or row.b_id in used:
            continue
        chosen.append(row._asdict())
        used |= {row.a_id, row.b_id}
        if len(chosen) == n:
            break
    out = pd.DataFrame(chosen, columns=list(p.columns)).drop(columns="order")
    out["a_token"] = [text[a] for a in out["a_id"]]
    out["b_token"] = [text[b] for b in out["b_id"]]
    out["tier_label"] = out["tier"].map(TIER_LABELS)
    return out.reset_index(drop=True)
