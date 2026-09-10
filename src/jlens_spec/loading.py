"""Clean-pass workspace loading: cos(h, v_tok) and rank, per position/layer/token."""
from __future__ import annotations

import pandas as pd
import torch
import torch.nn.functional as F

from . import lens as lens_mod
from . import model as model_mod


def loading(model, lens, prompt, token_ids: list[int], layers: list[int]) -> pd.DataFrame:
    """One clean forward pass; all `layers` read from the same trace. Columns: stimulus_id,
    question_key, pos, class, layer, token, token_id, cos, rank."""
    saved = {}
    with model.trace(prompt.input_ids):
        for l in layers:
            saved[l] = model_mod.layer_output(model, l).float().save()

    tokenizer = model.tokenizer
    rows = []
    for l in layers:
        h = saved[l][0]  # [pos, d]
        for tok_id in token_ids:
            v = lens_mod.lens_vectors(model, lens, [tok_id], l)[0]
            cos = F.cosine_similarity(h, v.unsqueeze(0), dim=-1)  # [pos]
            rank = lens_mod.token_rank(model, lens, h, l, tok_id)  # [pos]
            token_str = tokenizer.decode([tok_id])
            for pos in range(h.shape[0]):
                rows.append(
                    {
                        "stimulus_id": prompt.stimulus_id,
                        "question_key": prompt.question_key,
                        "pos": pos,
                        "class": prompt.classes[pos],
                        "layer": l,
                        "token": token_str,
                        "token_id": tok_id,
                        "cos": float(cos[pos]),
                        "rank": int(rank[pos]),
                    }
                )
    return pd.DataFrame(rows)


def single_token_pairs(tokenizer, pairs: dict) -> tuple[dict, list[str]]:
    """Keep only pairs whose both members are single tokens under `tokenizer`. Returns
    (kept, dropped_names). Used at runtime instead of rewriting configs/tokens.yaml."""
    kept, dropped = {}, []
    for name, (a, b) in pairs.items():
        ok = all(len(tokenizer.encode(t, add_special_tokens=False)) == 1 for t in (a, b))
        if ok:
            kept[name] = [a, b]
        else:
            dropped.append(name)
    return kept, dropped


def _question_hidden(model, prompt, layers: list[int]) -> tuple[list[int], dict]:
    q_pos = [i for i, c in enumerate(prompt.classes) if c == "question"]
    saved = {}
    with model.trace(prompt.input_ids):
        for l in layers:
            saved[l] = model_mod.layer_output(model, l).float()[0, q_pos].save()
    return q_pos, saved


def question_topk_ids(model, lens, prompt, layers: list[int], k: int = 25) -> set[int]:
    """Union of the lens readout's top-k token ids over every question-class position and every
    layer in `layers` -- the candidate pool for control-token selection (M2 step 1)."""
    _, saved = _question_hidden(model, prompt, layers)
    ids: set[int] = set()
    for l in layers:
        top_ids, _ = lens_mod.readout(model, lens, saved[l], l, k=k)
        ids.update(top_ids.flatten().tolist())
    return ids


def question_token_stats(model, lens, prompt, token_ids: list[int], layers: list[int]) -> pd.DataFrame:
    """Per token in `token_ids`: cos(h, v_tok) and full-readout rank, averaged over every
    question-class position x layer of this prompt (aggregated here, since per-position rows for
    thousands of candidates would be tens of millions of rows). Columns: stimulus_id, question_key,
    token_id, token, mean_cos, mean_rank."""
    _, saved = _question_hidden(model, prompt, layers)
    tokenizer = model.tokenizer
    cos_sum = rank_sum = None
    n = 0
    for l in layers:
        h = saved[l]  # [Q, d]
        V = lens_mod.lens_vectors(model, lens, token_ids, l)  # [T, d]
        cos = F.normalize(h, dim=-1) @ F.normalize(V, dim=-1).T  # [Q, T], no [Q, T, d] intermediate
        logits = lens_mod._transport_unembed(model, lens, h, l).float()  # [Q, vocab]
        vals = logits[:, torch.tensor(token_ids, device=logits.device)].contiguous()  # [Q, T]
        sorted_asc, _ = torch.sort(logits, dim=-1)
        # 1 + #(logits > val) == vocab - #(logits <= val) + 1
        rank = (logits.shape[-1] - torch.searchsorted(sorted_asc, vals, right=True) + 1).float()
        cos_sum = cos.sum(0) if cos_sum is None else cos_sum + cos.sum(0)
        rank_sum = rank.sum(0) if rank_sum is None else rank_sum + rank.sum(0)
        n += h.shape[0]
    return pd.DataFrame({
        "stimulus_id": prompt.stimulus_id,
        "question_key": prompt.question_key,
        "token_id": token_ids,
        "token": [tokenizer.decode([t]) for t in token_ids],
        "mean_cos": (cos_sum / n).tolist(),
        "mean_rank": (rank_sum / n).tolist(),
    })


def _matrix_lang_of(stimulus_id: str) -> str | None:
    """Derived from the stimuli.json naming convention verified in stimuli/stimuli.json: ids are
    prefixed 'sp_' for es-matrix passages and 'fr_' for fr-matrix passages."""
    if stimulus_id.startswith("sp_"):
        return "es"
    if stimulus_id.startswith("fr_"):
        return "fr"
    return None


def pair_score(df: pd.DataFrame, pair: tuple[str, str], band: list[int]) -> float:
    """min(mean cos of the Spanish member over Spanish-language positions,
           mean cos of the French member over French-language positions), over `band`.
    Spanish-language positions = matrix positions of es-matrix passages + intrusion positions of
    fr-matrix passages (symmetric for French)."""
    es_tok, fr_tok = pair
    d = df[df["layer"].isin(band)].copy()
    d["matrix_lang"] = d["stimulus_id"].map(_matrix_lang_of)

    es_positions = ((d["matrix_lang"] == "es") & (d["class"] == "matrix")) | (
        (d["matrix_lang"] == "fr") & (d["class"] == "intrusion")
    )
    fr_positions = ((d["matrix_lang"] == "fr") & (d["class"] == "matrix")) | (
        (d["matrix_lang"] == "es") & (d["class"] == "intrusion")
    )

    es_mean = d.loc[es_positions & (d["token"] == es_tok), "cos"].mean()
    fr_mean = d.loc[fr_positions & (d["token"] == fr_tok), "cos"].mean()
    # Python's min() with a NaN depends on argument order; make "no data" explicit instead.
    if es_mean != es_mean or fr_mean != fr_mean:
        return float("nan")
    return float(min(es_mean, fr_mean))
