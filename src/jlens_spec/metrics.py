"""Logprob-margin metrics. Never sampled binary answers -- always full-distribution margins.

CALL CONVENTION NOTE: `question_margin`'s contract is `(question_key, logprobs, tokens_cfg,
matrix_lang)` with no tokenizer, but resolving "Spanish"/"French"/"Hola"/"Bonjour" etc. to token ids
needs one. `build_tokens_cfg` (below) is the intended way to produce `tokens_cfg`: called once per
model/tokenizer with the raw configs/tokens.yaml dict, it returns id-lists that `question_margin`
and `argmax_label` consume directly.
"""
from __future__ import annotations

import torch


def margin(logprobs: torch.Tensor, pos_ids: list[int], neg_ids: list[int]) -> float:
    """logsumexp(logprobs[pos_ids]) - logsumexp(logprobs[neg_ids])."""
    pos = torch.logsumexp(logprobs[pos_ids], dim=0)
    neg = torch.logsumexp(logprobs[neg_ids], dim=0)
    return float(pos - neg)


def answer_ids(tokenizer, answers: list[str]) -> list[int]:
    """First token id of each answer string, tried with and without a leading space, deduplicated
    (order-preserving)."""
    ids: list[int] = []
    seen: set[int] = set()
    for a in answers:
        for variant in (a, " " + a):
            enc = tokenizer.encode(variant, add_special_tokens=False)
            if not enc:
                continue
            tid = enc[0]
            if tid not in seen:
                seen.add(tid)
                ids.append(tid)
    return ids


def build_tokens_cfg(tokenizer, raw: dict) -> dict:
    """Convert the raw configs/tokens.yaml dict (strings) into id-lists for question_margin /
    argmax_label. `raw` must have "answers" (yes/no/hello) and "language_tokens" (es/fr forms)."""
    yes_ids = answer_ids(tokenizer, raw["answers"]["yes"])
    no_ids = answer_ids(tokenizer, raw["answers"]["no"])

    lang_ids: dict[str, list[int]] = {}
    for lang, forms in raw["language_tokens"].items():
        ids: list[int] = []
        for f in forms:
            enc = tokenizer.encode(f, add_special_tokens=False)
            if enc:
                ids.append(enc[0])
        lang_ids[lang] = sorted(set(ids))

    hello_ids = {lang: answer_ids(tokenizer, forms) for lang, forms in raw["answers"]["hello"].items()}

    return {"yes_ids": yes_ids, "no_ids": no_ids, "lang_ids": lang_ids, "hello_ids": hello_ids}


def _dispatch(question_key: str, tokens_cfg: dict, matrix_lang: str):
    """Return (pos_ids, neg_ids, pos_label, neg_label). pos = 'answer as if the label were the
    *target* (intrusion) language' for report/hello; pos = Yes for anomaly/content."""
    target_lang = "fr" if matrix_lang == "es" else "es"
    if question_key in ("anomaly", "content"):
        return tokens_cfg["yes_ids"], tokens_cfg["no_ids"], "yes", "no"
    if question_key == "report":
        return (
            tokens_cfg["lang_ids"][target_lang],
            tokens_cfg["lang_ids"][matrix_lang],
            "target",
            "source",
        )
    if question_key == "hello":
        return (
            tokens_cfg["hello_ids"][target_lang],
            tokens_cfg["hello_ids"][matrix_lang],
            "target",
            "source",
        )
    raise ValueError(f"unknown question_key {question_key!r}")


def question_margin(question_key: str, logprobs: torch.Tensor, tokens_cfg: dict, matrix_lang: str) -> float:
    pos_ids, neg_ids, _, _ = _dispatch(question_key, tokens_cfg, matrix_lang)
    return margin(logprobs, pos_ids, neg_ids)


def argmax_label(question_key: str, logprobs: torch.Tensor, tokens_cfg: dict, matrix_lang: str) -> str:
    """Which side of the question's answer set has the higher max logprob: 'yes'/'no' for
    anomaly/content, 'target'/'source' for report/hello. Used to compute `flip` in runner.py."""
    pos_ids, neg_ids, pos_label, neg_label = _dispatch(question_key, tokens_cfg, matrix_lang)
    pos_best = max(float(logprobs[i]) for i in pos_ids)
    neg_best = max(float(logprobs[i]) for i in neg_ids)
    return pos_label if pos_best >= neg_best else neg_label


def topk_tokens(logits: torch.Tensor, tokenizer, k: int) -> list[dict]:
    """The k highest-scoring tokens of a 1-D logit vector, best first, as plain dicts
    {"token_id", "token", "logit", "logprob"} (storable in parquet). How many to SAVE is the run's
    `save_topk` setting; figures and summaries cut this list down to whatever k they show."""
    logits = logits.float()
    logprobs = torch.log_softmax(logits, dim=-1)
    vals, ids = torch.topk(logits, min(k, logits.shape[-1]))
    return [{"token_id": int(i), "token": tokenizer.decode([int(i)]), "logit": float(v),
             "logprob": float(logprobs[i])} for v, i in zip(vals, ids)]
