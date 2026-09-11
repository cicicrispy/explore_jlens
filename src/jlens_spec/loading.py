"""Clean-pass workspace loading: cos(h, v_tok) and rank, per position/layer/token."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import lens as lens_mod
from . import model as model_mod


def prompt_pass(model, lens, prompt, lang_token_ids: list[int], save_k: int):
    """M2's clean pass for ONE prompt: one forward pass, every lens layer read from the same trace.

    Returns (loadings, topk, answer_logits):
      loadings: DataFrame, one row per (position, lens layer, language token) -- columns
          stimulus_id, question_key, pos, class, layer, token, token_id, cos, rank. cos = cos(h, v_tok)
          in float32; rank = 1-based rank of the token in the full lens readout at that position.
      topk: pyarrow Table, one row per (position, lens layer) -- columns stimulus_id, question_key,
          pos, class, layer, topk_ids (int32 x save_k), topk_logits (float32 x save_k): the lens
          readout's top-`save_k` tokens, best first.
      answer_logits: Tensor[vocab], the model's own logits at metric_pos (where the answer is read).
    Every position is included (template tokens too); every lens layer (none is skipped)."""
    import pyarrow as pa

    layers = list(lens.layers)
    # A plain loop, not a comprehension: inside an nnsight trace a comprehension's results never get
    # assigned.
    saved = {}
    with model.trace(prompt.input_ids):
        for l in layers:
            saved[l] = model_mod.layer_output(model, l).float()[0].save()
        answer = model.output.logits[0, prompt.metric_pos].float().save()

    tok = model.tokenizer
    n = len(prompt.input_ids)
    texts = [tok.decode([t]) for t in lang_token_ids]
    classes = np.array(prompt.classes, dtype=object)
    load_parts, top_ids, top_logits = [], [], []
    for l in layers:
        h = saved[l]                                                    # [pos, d]
        logits = lens_mod._transport_unembed(model, lens, h, l).float()  # [pos, vocab]
        V = lens_mod.lens_vectors(model, lens, lang_token_ids, l).to(h.device)  # [T, d]
        cos = (F.normalize(h, dim=-1) @ F.normalize(V, dim=-1).T).cpu().numpy()  # [pos, T]
        idx = torch.tensor(lang_token_ids, device=logits.device)
        vals = logits[:, idx]                                           # [pos, T]
        rank = torch.stack([1 + (logits > vals[:, j:j + 1]).sum(-1) for j in range(len(lang_token_ids))],
                           dim=-1).cpu().numpy()                         # [pos, T]
        T = len(lang_token_ids)
        load_parts.append(pd.DataFrame({
            "pos": np.repeat(np.arange(n), T), "class": np.repeat(classes, T), "layer": int(l),
            "token": np.tile(texts, n), "token_id": np.tile(lang_token_ids, n),
            "cos": cos.reshape(-1).astype(np.float32), "rank": rank.reshape(-1).astype(np.int64)}))
        tv, ti = torch.topk(logits, save_k, dim=-1)
        top_ids.append(ti.to(torch.int32).cpu().numpy())
        top_logits.append(tv.cpu().numpy().astype(np.float32))

    loadings = pd.concat(load_parts, ignore_index=True)
    loadings.insert(0, "question_key", prompt.question_key)
    loadings.insert(0, "stimulus_id", prompt.stimulus_id)

    ids = np.concatenate(top_ids)          # [layers * pos, k], layer-major
    vals = np.concatenate(top_logits)
    n_rows = ids.shape[0]
    topk = pa.table({
        "stimulus_id": pa.array([prompt.stimulus_id] * n_rows),
        "question_key": pa.array([prompt.question_key] * n_rows),
        "pos": pa.array(np.tile(np.arange(n), len(layers)).astype(np.int32)),
        "class": pa.array(list(classes) * len(layers)),
        "layer": pa.array(np.repeat(np.array(layers, dtype=np.int32), n)),
        "topk_ids": pa.FixedSizeListArray.from_arrays(pa.array(ids.reshape(-1)), save_k),
        "topk_logits": pa.FixedSizeListArray.from_arrays(pa.array(vals.reshape(-1)), save_k),
    })
    return loadings, topk, answer


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
