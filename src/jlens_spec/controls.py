"""Control-token selection for M3 -- picked by rule, then read by the human before any cell runs
(the spec's "propose, then the human approves" step, in this form since 2026-09-11; see README
"Departures from the spec").

Controls are picked by a CONTROLS run (scripts/m3_controls.py, one per position set) and saved in its
controls/ folder; the run then stops. The human reads its summary; the only lever is
`controls.control_ineligible` in configs/tokens.yaml -- extend it, commit, and run the controls step
again (a new run folder). The position set's POSITIVES run names the accepted controls run
(`controls_run:` in its experiment file) and copies its selection.yaml; the ANOMALY run reuses the
positives run's copy. Both refuse a selection that doesn't match them (selection_problems).

Each treatment swap exchanges the language being REMOVED (s) with the one SWAPPED IN (t). With the
treatment pair (es word, fr word) there are four treatment rows per position set:
    passage es, m2i: s = es word, t = fr word        passage es, i2m: s = fr word, t = es word
    passage fr, m2i: s = fr word, t = es word        passage fr, i2m: s = es word, t = fr word
(runner._resolve_pair_words is the same mapping.) With four questions (the positives run uses
report + hello, the anomaly run anomaly + content) that makes 4 x 4 = 16 CHECKS ("report_es_m2i",
...); a check's prompts are the 8 passages of that row's passage language, asked that question. The
label_to_present controls are picked FOR EACH CHECK SEPARATELY (a cell uses the controls of its own
check); the big_nonlabel controls are ONE set shared by every check, so their rules must hold in
each of the 16.

Measures -- always within the run's band only, at the positions the treatment edits:
- layer coverage of token x on one prompt = the fraction of the band's layers at which x is among the
  lens readout's top-`pool_k` tokens at >= 1 edited position (from M2's saved top-100 readout). A
  check's value = the mean over its prompts.
- |Δc| of a swap between tokens s and x at one position and layer = ||flip(c) - c|| with
  c = pinv([v_s, v_x]) h -- the same quantity M3 logs as delta_c_norm -- here from a CLEAN forward
  pass (M3 then measures the real value). A check's value = the mean over its prompts x edited
  positions x band layers -- the same average the M3 summary reports as the measured ratio.

Candidates: every token in the top-`pool_k` readout at the edited positions within the band, on any
of the 64 prompts, minus
  - `control_ineligible` tokens and any token whose text contains one of them (case-insensitive);
  - the answer tokens (every form metrics.answer_ids gives: Yes / No / the hello words, with and
    without a leading space) -- swapping one in could push an answer directly;
  - every token without a letter or a digit (punctuation, blank lines, symbols) -- odd tokens that
    can have large, unspecific effects;
  - any token that occurs in the passages' own text: its text, without leading spaces and ignoring
    case, equals that of a token with at least one letter at a matrix or intrusion position of any
    prompt (passage_words) -- a word of either language would put a language back in.
Tokenizer special tokens and tokens whose text does not re-tokenize to exactly themselves are never
used as controls; they are listed separately (`excluded_special`).

label_to_present -- keeps the removed label s; the control token x stands in for the swapped-in t,
    and must LOWER the label: the swap exchanges coordinates, so the label's coordinate c_s becomes
    c_x, which lowers it only if c_x < c_s. Per check: x qualifies only if the mean of (c_s - c_x)
    over the check's prompts x edited positions x band layers is positive (the same average as
    below); among those, coverage ratio = cov(x) / cov(t) and |Δc| ratio = |Δc|(s, x) / |Δc|(s, t)
    set the tier. A token that does not lower the label is used only if a check has too few that
    do, flagged "does not lower the label".
big_nonlabel -- a control pair (a, b), a standing in for s and b for t. The swap is symmetric and the
    rows swap the roles of the two language tokens, so each member must reach the bar against both:
    coverage ratio of a member = cov(member) / max(cov(s), cov(t)), in every check. The edit is
    scaled to norm_scale x the treatment's ||Δh|| at every position and layer, and after that
    scaling its |Δc| no longer depends on the state h at all: at each layer it is
    norm_scale x (||v_s - v_t|| / ||v_a - v_b||) x the treatment's |Δc| there. So the pair's |Δc|
    ratio (the |Δc| it will actually have, over the treatment's, averaged as above) comes from the
    clean pass's treatment |Δc| per layer and the lens vectors -- no forward pass per pair. Pairs
    are formed from the `pool_size` members with the widest coverage; the chosen pairs share no token.

Tiers -- selection never stops:
    1 = every ratio (coverage and |Δc|) >= bars[0] (100%) -- in its check (label_to_present) or in
    every check (big_nonlabel);  2 = every ratio >= bars[1] (75%);  3 = the closest remaining,
    flagged "below 75%";  4 (label_to_present only) = does not lower the label, flagged.
Within tiers 1-2 the CLOSEST to the treatment is preferred: the smallest |log(|Δc| ratio)| (for
big_nonlabel its mean over the checks). Tier 3 is ordered by its smallest ratio, largest first.
A treatment token that is never in the top-`pool_k` in a check makes that coverage ratio infinite:
any candidate has at least that much coverage, so coverage sets no bar there.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch

from . import lens as lens_mod
from . import model as model_mod

ROWS = ("es_m2i", "es_i2m", "fr_m2i", "fr_i2m")
TIER_LABELS = {1: ">=100%", 2: ">=75%", 3: "below 75%", 4: "does not lower the label"}
# controls/selection.yaml layout and rules; a positives or anomaly run refuses a file with another
# version (4: answer tokens and passage words excluded; picked by a separate controls run).
SELECTION_VERSION = 4
# The settings a run must share with the selection it uses (selection_problems).
MATCH_KEYS = ("position_set", "band_layers", "from_m2_run", "pair_words", "skip_first")
PASSAGE_CLASSES = ("matrix", "intrusion")
NO_LETTER_OR_DIGIT = "punctuation (no letter or digit)"   # an excluded_reason


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


def group_name(matrix_lang: str, question: str) -> str:
    """The prompts of one passage language asked one question, e.g. 'es_report'."""
    return f"{matrix_lang}_{question}"


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


def coverage_by_group(cov: pd.DataFrame, prompts_by_group: dict, n_band_layers: int) -> pd.DataFrame:
    """Mean coverage fraction per group of prompts: rows = token_id, one column cov_<group> per
    group. A token absent from a prompt counts 0 for that prompt. `prompts_by_group` = {"es_report":
    [(stimulus_id, question_key), ...], ...} (any group names)."""
    out = {}
    for group, keys in prompts_by_group.items():
        keyset = {f"{s}\t{q}" for s, q in keys}
        sub = cov[(cov["stimulus_id"] + "\t" + cov["question_key"]).isin(keyset)]
        out[f"cov_{group}"] = sub.groupby("token_id")["n_layers"].sum() / (n_band_layers * len(keys))
    return pd.DataFrame(out).fillna(0.0)


# ----------------------------------------------------------------------------- candidates


def word_key(text: str) -> str:
    """A token's text for the passage-word rule: without leading spaces, lower case (' Sous' -> 'sous')."""
    return text.lstrip(" ").lower()


def passage_words(prompts, tokenizer) -> set[str]:
    """word_key of every token with at least one letter at a matrix or intrusion position of these
    prompts -- the passages' own words, in any case and with or without a leading space. Tokens
    without a letter are left out here (punctuation has its own rule in classify_tokens)."""
    out = set()
    for p in prompts:
        for tid, cls in zip(p.input_ids, p.classes):
            if cls in PASSAGE_CLASSES:
                key = word_key(tokenizer.decode([int(tid)]))
                if any(ch.isalpha() for ch in key):
                    out.add(key)
    return out


def classify_tokens(token_ids, tokenizer, ineligible, answer_ids, passage_words) -> pd.DataFrame:
    """For each id: its text and why it can't be a control ('' = eligible). Special tokens and tokens
    that don't re-tokenize to themselves are kept apart ('special token' / 'does not re-tokenize to
    itself') -- never controls, listed separately. `answer_ids`: the answer tokens' ids
    (metrics.answer_ids); `passage_words`: see passage_words()."""
    special = set(tokenizer.all_special_ids) | set(getattr(tokenizer, "added_tokens_decoder", {}) or {})
    names = [n.strip().lower() for n in ineligible if n.strip()]
    ineligible, answer_ids, passage_words = set(ineligible), {int(i) for i in answer_ids}, set(passage_words)
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
        elif tid in answer_ids:
            reason = "answer token"
        elif not any(ch.isalnum() for ch in text):
            reason = NO_LETTER_OR_DIGIT
        elif word_key(text) in passage_words:
            reason = "occurs in the passages"
        else:
            reason = ""
        rows.append({"token_id": tid, "token": text, "excluded_reason": reason})
    return pd.DataFrame(rows, columns=["token_id", "token", "excluded_reason"])


def selection_problems(selection: dict, **expected) -> list[str]:
    """Why a controls/selection.yaml can't be used by a run with these settings ([] = it can): it must
    be written by the current rules (SELECTION_VERSION) and match the run on every MATCH_KEYS setting,
    all of which must be given. The code version that picked it (recorded as git_commit) is NOT
    compared: the controls may be picked again after M2 while the code changes for other reasons, and
    a change of the rules themselves bumps SELECTION_VERSION."""
    missing = [k for k in MATCH_KEYS if k not in expected]
    if missing:
        raise ValueError(f"selection_problems needs {missing}")
    if selection.get("selection_version") != SELECTION_VERSION:
        return [f"selection_version {selection.get('selection_version')} -- picked by other rules "
                f"(this code expects {SELECTION_VERSION})"]
    return [f"{k}: {selection.get(k)!r} there, {expected[k]!r} here" for k in MATCH_KEYS
            if selection.get(k) != expected[k]]


def coverage_text(ratio: float) -> str:
    return "coverage: no bar (the treatment tokens are never in the top-100 here)" if ratio == float("inf") \
        else f"coverage {ratio:.2f}x"


def control_lines(selection: dict, questions) -> list[str]:
    """The picked controls, for a log or a summary: label_to_present one line per check of these
    questions (each check has its own tokens), big_nonlabel one line per pair (shared by every check)."""
    n = len(selection["checks"])
    cov = coverage_text
    lines = []
    for k, t in selection["checks"].items():
        if t["question"] not in questions:
            continue
        picks = "; ".join(f"{c['token']!r} ({c['tier_label']}, |Δc| {c['dc_ratio']:.2f}x, {cov(c['cov_ratio'])})"
                          for c in selection["label_to_present"][k])
        lines.append(f"  - label_to_present, {k}: {picks}")
    lines += [f"  - big_nonlabel[{c['control_index']}]: {c['a_token']!r} <-> {c['b_token']!r}, tier {c['tier_label']} "
              f"-- lowest over the {n} checks: |Δc| after scaling {c['dc_ratio_worst']:.3f}x the treatment's, "
              f"{cov(c['cov_ratio_worst'])}" for c in selection["big_nonlabel"]]
    return lines


def big_nonlabel_pool(cands: pd.DataFrame, treat: dict, pool_size: int, bars) -> pd.DataFrame:
    """The `pool_size` big_nonlabel members with the best coverage tier, then the widest mean
    coverage. `cands`: eligible candidates with cov_<check> columns; `treat[check]` has cov_removed
    and cov_swapped_in. Adds columns member_worst_cov_ratio and member_tier."""
    ratios = []
    for check, t in treat.items():
        ref = max(t["cov_removed"], t["cov_swapped_in"])
        ratios.append(np.full(len(cands), np.inf) if ref == 0 else cands[f"cov_{check}"].to_numpy() / ref)
    worst = np.min(np.stack(ratios), axis=0) if len(cands) else np.array([])
    out = cands.assign(member_worst_cov_ratio=worst, member_tier=_tier(worst, bars),
                       mean_cov=cands[[f"cov_{check}" for check in treat]].mean(axis=1))
    return out.sort_values(["member_tier", "mean_cov"], ascending=[True, False]).head(pool_size)


# ------------------------------------------------------------------------- |Δc| estimates


def _signed_diff(a, b, c, x, y):
    """c_s - c_x for the swap basis V = [v_s, v_x], from dot products: a = v_s.v_s, b = v_s.v_x,
    c = v_x.v_x, x = h.v_s, y = h.v_x (c = pinv(V) h = (V^T V)^-1 V^T h). Positive = the swap LOWERS
    the coordinate of s (it becomes c_x). Broadcasts."""
    return ((c + b) * x - (a + b) * y) / (a * c - b * b)


def _abs_dc(a, b, c, x, y):
    """|Δc| = sqrt(2) |c_s - c_x| (see _signed_diff). Broadcasts."""
    return math.sqrt(2.0) * torch.abs(_signed_diff(a, b, c, x, y))


def estimate_delta_c(model, lens, prompts, masks, band_layers, cand_ids, pair_ids, pool_ids) -> dict:
    """Clean-pass |Δc| estimates at each prompt's edited positions x band layers, plus the lens-vector
    distances the big_nonlabel rule needs.

    prompts/masks: parallel lists (Prompt, bool mask over its positions). cand_ids: label_to_present
    candidates. pair_ids: (es word id, fr word id) of the treatment pair. pool_ids: big_nonlabel pool.
    Returns (per-prompt arrays have one row per prompt, in `prompts` order):
      ltp_removed_es / ltp_removed_fr: [prompts, cands] -- mean |Δc|(removed label, candidate)
      ltp_lowering_es / ltp_lowering_fr: [prompts, cands] -- mean (c_label - c_candidate); positive =
          the swap lowers the removed label's coordinate
      treat: [prompts] -- mean |Δc|(es word, fr word), the treatment swap
      treat_es_minus_fr: [prompts] -- mean (c_es - c_fr) in the treatment's basis
      treat_by_layer: [prompts, band layers] -- the treatment's |Δc| SUMMED over the edited
          positions at each band layer (treat = its row sum / n_positions_x_layers)
      n_positions_x_layers: [prompts]
      treat_dist: [band layers] -- ||v_es - v_fr|| at each band layer
      pairs: list of (i, j) index pairs into pool_ids;  pair_dist: [pairs, band layers] -- ||v_a - v_b||
    A candidate (nearly) parallel to the label gives a division by ~0 -> inf/NaN; callers drop those."""
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

    n_p, n_c, n_l = len(prompts), len(cand_ids), len(band_layers)
    pairs = [(i, j) for i in range(len(pool_ids)) for j in range(i + 1, len(pool_ids))]
    acc = {"es": np.zeros((n_p, n_c)), "fr": np.zeros((n_p, n_c)), "es_lower": np.zeros((n_p, n_c)),
           "fr_lower": np.zeros((n_p, n_c)), "treat_by_layer": np.zeros((n_p, n_l)), "treat_signed": np.zeros(n_p)}
    counts = np.zeros(n_p)
    treat_dist = np.zeros(n_l)
    pair_dist = np.zeros((len(pairs), n_l))
    pi = torch.tensor([i for i, _ in pairs], dtype=torch.long)
    pj = torch.tensor([j for _, j in pairs], dtype=torch.long)
    for li, l in enumerate(band_layers):
        Vc = lens_mod.lens_vectors(model, lens, list(cand_ids), l).to(dev) if n_c else None  # [T, d]
        ves, vfr = lens_mod.lens_vectors(model, lens, list(pair_ids), l).to(dev)
        a_es, a_fr, g = ves @ ves, vfr @ vfr, ves @ vfr
        treat_dist[li] = float(torch.sqrt(torch.clamp(a_es + a_fr - 2 * g, min=0.0)))
        if pairs:
            Vq = lens_mod.lens_vectors(model, lens, list(pool_ids), l).to(dev)             # [Q, d]
            G = Vq @ Vq.T
            pair_dist[:, li] = torch.sqrt(torch.clamp(G[pi, pi] + G[pj, pj] - 2 * G[pi, pj], min=0.0)).cpu().numpy()
        if Vc is not None:
            cc = (Vc * Vc).sum(-1)
            b_es, b_fr = Vc @ ves, Vc @ vfr
        for k in range(n_p):
            H = hs[k][l].to(dev)                                   # [P, d]
            if H.shape[0] == 0:
                continue
            x_es, x_fr = H @ ves, H @ vfr                          # [P]
            sd = _signed_diff(a_es, g, a_fr, x_es, x_fr)           # c_es - c_fr
            acc["treat_by_layer"][k, li] = float(math.sqrt(2.0) * sd.abs().sum())
            acc["treat_signed"][k] += float(sd.sum())
            if Vc is not None:
                Y = H @ Vc.T                                       # [P, T]
                sd_es = _signed_diff(a_es, b_es, cc, x_es[:, None], Y)   # c_es - c_x
                sd_fr = _signed_diff(a_fr, b_fr, cc, x_fr[:, None], Y)   # c_fr - c_x
                acc["es"][k] += (math.sqrt(2.0) * sd_es.abs()).sum(0).cpu().numpy()
                acc["fr"][k] += (math.sqrt(2.0) * sd_fr.abs()).sum(0).cpu().numpy()
                acc["es_lower"][k] += sd_es.sum(0).cpu().numpy()
                acc["fr_lower"][k] += sd_fr.sum(0).cpu().numpy()
            counts[k] += H.shape[0]
    denom = np.maximum(counts, 1)
    return {"ltp_removed_es": acc["es"] / denom[:, None], "ltp_removed_fr": acc["fr"] / denom[:, None],
            "ltp_lowering_es": acc["es_lower"] / denom[:, None], "ltp_lowering_fr": acc["fr_lower"] / denom[:, None],
            "treat": acc["treat_by_layer"].sum(axis=1) / denom, "treat_by_layer": acc["treat_by_layer"],
            "treat_es_minus_fr": acc["treat_signed"] / denom,
            "n_positions_x_layers": counts, "treat_dist": treat_dist, "pairs": pairs, "pair_dist": pair_dist}


def with_coverage(cands: pd.DataFrame, cov_group: pd.DataFrame, pair_words, pair_ids,
                  questions) -> tuple[pd.DataFrame, dict]:
    """Step 1 (no model): per-check coverage. Adds cov_<check> to `cands` (token_id, token, ...) and
    returns treat = {check: {row, question, group, matrix_lang, direction, removed, swapped_in,
    cov_removed, cov_swapped_in}}, one check per question x row ("report_es_m2i", ...)."""
    word_id = dict(zip(pair_words, pair_ids))

    def cov_of(ids, group):
        col = f"cov_{group}"
        if col not in cov_group:
            return np.zeros(len(ids))
        return cov_group[col].reindex(ids).fillna(0.0).to_numpy(dtype=float)

    treat, cols = {}, {}
    for q in questions:
        for row in treatment_rows(pair_words):
            check, group = f"{q}_{row['row']}", group_name(row["matrix_lang"], q)
            treat[check] = {**row, "question": q, "group": group,
                            "cov_removed": float(cov_of([word_id[row["removed"]]], group)[0]),
                            "cov_swapped_in": float(cov_of([word_id[row["swapped_in"]]], group)[0])}
            cols[f"cov_{check}"] = cov_of(cands["token_id"].to_numpy(), group)
    return pd.concat([cands.reset_index(drop=True), pd.DataFrame(cols)], axis=1), treat


def with_delta_c(cands: pd.DataFrame, treat: dict, est: dict, groups: list[str], pair_words,
                 pool_ids, norm_scale: float) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Step 2 (after estimate_delta_c): per-check |Δc|. `cands` must be in the order `est` was computed
    for; `groups` = each estimated prompt's group name (group_name(passage language, question)).
    Adds dc_<check> (the label_to_present swap's |Δc|) and lower_<check> (mean c_label - c_candidate;
    positive = it lowers the removed label) to `cands`, and `dc` and `lowers_by` (the treatment's own
    mean c_removed - c_swapped_in, for the record) to each treat check; returns the big_nonlabel pair
    table (a_id, b_id, dc_<check>), where dc is the |Δc| the pair's swap will have AFTER it is
    scaled to norm_scale x the treatment's ||Δh|| (module docstring)."""
    es_word = pair_words[0]
    groups = np.asarray(groups)
    treat = {k: dict(v) for k, v in treat.items()}
    pairs = pd.DataFrame({"a_id": [int(pool_ids[i]) for i, _ in est["pairs"]],
                          "b_id": [int(pool_ids[j]) for _, j in est["pairs"]]}, dtype=np.int64)
    denom = np.maximum(est["n_positions_x_layers"], 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = est["treat_dist"][None, :] / est["pair_dist"]                         # [pairs, layers]
        pair_mean = norm_scale * (est["treat_by_layer"] @ ratio.T) / denom[:, None]  # [prompts, pairs]
    c_cols, p_cols = {}, {}
    for check, t in treat.items():
        idx = groups == t["group"]
        if not idx.any():
            raise ValueError(f"no estimated prompt belongs to {t['group']} (check {check})")
        removes_es = t["removed"] == es_word
        t["dc"] = float(est["treat"][idx].mean())
        t["lowers_by"] = float((1 if removes_es else -1) * est["treat_es_minus_fr"][idx].mean())
        c_cols[f"dc_{check}"] = est["ltp_removed_es" if removes_es else "ltp_removed_fr"][idx].mean(axis=0) \
            if len(cands) else np.zeros(0)
        c_cols[f"lower_{check}"] = est["ltp_lowering_es" if removes_es else "ltp_lowering_fr"][idx].mean(axis=0) \
            if len(cands) else np.zeros(0)
        p_cols[f"dc_{check}"] = pair_mean[idx].mean(axis=0) if len(pairs) else np.zeros(0)
    cands = pd.concat([cands.reset_index(drop=True), pd.DataFrame(c_cols)], axis=1)
    pairs = pd.concat([pairs, pd.DataFrame(p_cols)], axis=1)
    return cands, treat, pairs


# ------------------------------------------------------------------------------ selection


def _rank(df: pd.DataFrame, cov_ratio_worst, dc_ratio_cols: list[str], bars) -> pd.DataFrame:
    """Shared by both kinds: drop non-finite |Δc| ratios, then worst ratios, tier and closeness, sorted
    best first (tiers 1-2 by closeness, tier 3 by its smallest ratio, largest first)."""
    df = df.assign(cov_ratio_worst=cov_ratio_worst)
    df = df[np.isfinite(df[dc_ratio_cols].to_numpy()).all(axis=1)].copy()
    df["dc_ratio_worst"] = df[dc_ratio_cols].min(axis=1)
    df["worst_ratio"] = np.minimum(df["cov_ratio_worst"], df["dc_ratio_worst"])
    df["tier"] = _tier(df["worst_ratio"].to_numpy(), bars)
    df["closeness"] = np.abs(np.log(df[dc_ratio_cols].clip(lower=1e-12))).mean(axis=1)
    df["order"] = np.where(df["tier"] < 3, df["closeness"], -df["worst_ratio"])
    return df.sort_values(["tier", "order"])


def select_label_to_present(cands: pd.DataFrame, treat: dict, n: int, bars) -> dict[str, pd.DataFrame]:
    """For EACH check separately, the top `n` label_to_present candidates (see the module
    docstring): only those that lower the removed label in that check, ranked by that check's
    coverage and |Δc| tiers, closest first; tier 4 (does not lower the label) only fills a check
    that has fewer than `n` that do. `cands` columns: token_id, token, cov_<check>, dc_<check>,
    lower_<check>; `treat[check]` has cov_swapped_in and dc. Returns {check: DataFrame with token_id,
    token, cov, dc, lowers_by, cov_ratio, dc_ratio, worst_ratio, tier, tier_label, closeness}."""
    out = {}
    for check, t in treat.items():
        ref = t["cov_swapped_in"]
        c = pd.DataFrame({"token_id": cands["token_id"].to_numpy(), "token": cands["token"].to_numpy(),
                          "cov": cands[f"cov_{check}"].to_numpy(dtype=float),
                          "dc": cands[f"dc_{check}"].to_numpy(dtype=float),
                          "lowers_by": cands[f"lower_{check}"].to_numpy(dtype=float)})
        c["cov_ratio"] = np.inf if ref == 0 else c["cov"] / ref
        c["dc_ratio"] = c["dc"] / t["dc"]
        r = _rank(c, c["cov_ratio"], ["dc_ratio"], bars).drop(columns=["cov_ratio_worst", "dc_ratio_worst"])
        r["base_tier"] = r["tier"]
        r["tier"] = np.where(r["lowers_by"] > 0, r["tier"], 4)
        r = r.sort_values(["tier", "base_tier", "order"]).head(n).drop(columns=["order", "base_tier"])
        r["tier_label"] = r["tier"].map(TIER_LABELS)
        out[check] = r.reset_index(drop=True)
    return out


def select_big_nonlabel(pool: pd.DataFrame, pair_dc: pd.DataFrame, treat: dict, n: int, bars) -> pd.DataFrame:
    """Pick `n` pairs that share no token (see the module docstring). `pool`: output of
    big_nonlabel_pool. `pair_dc`: columns a_id, b_id, dc_<check> (the |Δc| of the pair's swap after
    scaling). A pair's coverage ratio is the worse of its two members'."""
    worst = dict(zip(pool["token_id"], pool["member_worst_cov_ratio"]))
    text = dict(zip(pool["token_id"], pool["token"]))
    p = pair_dc.copy()
    for check, t in treat.items():
        p[f"dc_ratio_{check}"] = p[f"dc_{check}"] / t["dc"]
    cov_worst = [min(worst[a], worst[b]) for a, b in zip(p["a_id"], p["b_id"])]
    p = _rank(p, cov_worst, [f"dc_ratio_{check}" for check in treat], bars)
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


def never_in_top(treat: dict) -> dict[str, list[str]]:
    """{treatment token: ["es/report", ...]}: the passage-language/question groups where that token
    is never in the top-`pool_k` readout at the edited positions within the band (coverage 0) --
    there the coverage rule sets no bar, and the swap exchanges a token the lens does not read out."""
    out: dict[str, set] = {}
    for t in treat.values():
        for role in ("removed", "swapped_in"):
            if t[f"cov_{role}"] == 0:
                out.setdefault(t[role], set()).add(f"{t['matrix_lang']}/{t['question']}")
    return {w: sorted(g) for w, g in out.items()}
