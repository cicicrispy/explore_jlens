"""Prompt construction and per-token position classification.

CALL CONVENTION NOTE (flagged assumption, not literally specified): `build_prompt`'s contract is
`(stimulus, question_key, fmt)` with no explicit tokenizer/questions-map argument. Since both are
required to build anything, this implementation expects the caller to fold them into `fmt`:
    fmt = {**yaml_loaded_prompt_format, "tokenizer": tokenizer, "questions": stimuli_json["questions"]}
This is the only way to satisfy the 3-argument signature; flagged here for the human to confirm or
correct rather than silently inventing a 4th parameter. `fmt["user_message"]` (from
configs/prompt_format.yaml) is the user turn's text with {question} and {passage} placeholders --
the paper's wrapper -- see `make_fmt`.

Position classes: "question" (the question sentence), "matrix"/"intrusion" (the passage's
sentences), "instruction" (every other character of the user message: the paper's two instruction
sentences and the blank lines between the parts), and "template" (the chat template around the user
message: <|im_start|>user, <|im_end|>, the assistant/think prefix -- never edited).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

PositionClass = Literal["template", "instruction", "question", "matrix", "intrusion"]

# The classes a Stage-1 edit may target (template tokens never are). M3's two position sets:
#   question: the question sentence only;  message: every token of the user message.
POSITION_SETS = {
    "question": ("question",),
    "message": ("instruction", "question", "matrix", "intrusion"),
}

REQUIRED_SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
MANUAL_THINK_PREFILL = "<think>\n\n</think>\n\n"


@dataclass
class Prompt:
    text: str
    input_ids: list[int]
    classes: list[PositionClass]
    metric_pos: int
    stimulus_id: str
    question_key: str
    spans: dict
    flags: list[str] = field(default_factory=list)
    offsets: list[tuple[int, int]] = field(default_factory=list)  # char span of each token in `text`


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def make_fmt(tokenizer, stimuli: dict, prompt_format: dict) -> dict:
    """The `fmt` argument of build_prompt: configs/prompt_format.yaml's contents plus the tokenizer
    and stimuli.json's question strings."""
    return {**prompt_format, "tokenizer": tokenizer, "questions": stimuli["questions"]}


def fill_user_message(user_message: str, question: str, passage: str) -> tuple[str, int, int]:
    """Substitute the question and passage into `user_message` (which must contain "{question}"
    exactly once, followed later by "{passage}" exactly once). Returns (text, question_start,
    passage_start), the starts being character offsets within `text`. Plain string splitting, not
    str.format, so braces in a passage can never be misread as placeholders."""
    if user_message.count("{question}") != 1 or user_message.count("{passage}") != 1:
        raise ValueError(f"user_message must contain {{question}} and {{passage}} exactly once: {user_message!r}")
    before, rest = user_message.split("{question}")
    if "{passage}" not in rest:
        raise ValueError("user_message must put {question} before {passage}")
    middle, after = rest.split("{passage}")
    text = before + question + middle + passage + after
    return text, len(before), len(before) + len(question) + len(middle)


def build_prompt(stimulus: dict, question_key: str, fmt: dict) -> Prompt:
    tokenizer = fmt["tokenizer"]
    question_text = fmt["questions"][question_key]
    passage_text = stimulus["text"]
    content, q_rel, p_rel = fill_user_message(fmt["user_message"], question_text, passage_text)

    messages = [{"role": "user", "content": content}]

    flags: list[str] = []
    # HF passes unknown kwargs into the Jinja context silently, so a template that doesn't know
    # `enable_thinking` never raises -- detect the fallback case from the output instead.
    templated = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False, tokenize=False
    )
    if not templated.endswith(REQUIRED_SUFFIX) and templated.endswith("<|im_start|>assistant\n"):
        templated = templated + MANUAL_THINK_PREFILL
        flags.append("enable_thinking_unsupported_manual_prefill")

    assert templated.endswith(REQUIRED_SUFFIX), (
        f"templated prompt does not end with the required suffix {REQUIRED_SUFFIX!r}; "
        f"got tail: {templated[-80:]!r}"
    )

    enc = tokenizer(templated, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    suffix_ids = tokenizer.encode(REQUIRED_SUFFIX, add_special_tokens=False)
    assert input_ids[-len(suffix_ids):] == suffix_ids, (
        "tokens preceding metric_pos do not match the tokenized required suffix exactly"
    )

    c_start = templated.find(content)
    assert c_start != -1, "the user message was not found verbatim in the templated prompt"
    c_end = c_start + len(content)
    q_start = c_start + q_rel
    q_end = q_start + len(question_text)
    p_start = c_start + p_rel
    p_end = p_start + len(passage_text)
    assert templated[q_start:q_end] == question_text and templated[p_start:p_end] == passage_text
    # Everything in the user message that is neither the question nor the passage.
    instruction_spans = [(a, b) for a, b in ((c_start, q_start), (q_end, p_start), (p_end, c_end)) if b > a]

    sentence_spans = [
        {
            "role": s["role"],
            "char_start": p_start + s["char_start"],
            "char_end": p_start + s["char_end"],
        }
        for s in stimulus["sentences"]
    ]
    matrix_spans = [(s["char_start"], s["char_end"]) for s in sentence_spans if s["role"] == "matrix"]
    intrusion_spans = [(s["char_start"], s["char_end"]) for s in sentence_spans if s["role"] == "intrusion"]

    classes: list[PositionClass] = []
    for i, (s, e) in enumerate(offsets):
        if s == e:
            classes.append("template")
            continue
        q_ov = _overlap(s, e, q_start, q_end)
        m_ov = sum(_overlap(s, e, ms, me) for ms, me in matrix_spans)
        i_ov = sum(_overlap(s, e, ms, me) for ms, me in intrusion_spans)
        n_ov = sum(_overlap(s, e, a, b) for a, b in instruction_spans)
        # Ties go to the first entry (question/matrix/intrusion before instruction).
        overlaps = {"question": q_ov, "matrix": m_ov, "intrusion": i_ov, "instruction": n_ov}
        best_class, best_ov = max(overlaps.items(), key=lambda kv: kv[1])
        if best_ov == 0:
            classes.append("template")
            continue
        n_nonzero = sum(1 for v in overlaps.values() if v > 0)
        if n_nonzero > 1:
            flags.append(f"token_{i}_straddles_boundary")
        classes.append(best_class)  # type: ignore[arg-type]

    metric_pos = len(input_ids) - 1
    assert classes[metric_pos] == "template", (
        f"metric_pos {metric_pos} classified as {classes[metric_pos]!r}, expected 'template'"
    )

    spans = {
        "question": [q_start, q_end],
        "passage": [p_start, p_end],
        "sentences": sentence_spans,
        "instruction": [[a, b] for a, b in instruction_spans],
    }

    return Prompt(
        text=templated,
        input_ids=input_ids,
        classes=classes,
        metric_pos=metric_pos,
        stimulus_id=stimulus["id"],
        question_key=question_key,
        spans=spans,
        flags=flags,
        offsets=[(int(s), int(e)) for s, e in offsets],
    )


def region_check(prompt: Prompt) -> list[dict]:
    """For each region -- the question, every sentence, and every piece of instruction text --
    whether the tokens labelled with that region's class (and overlapping it) spell EXACTLY the
    region's text: nothing stripped, no extra characters. A token that also carries characters from
    outside its region (a separator newline, the space before a sentence, template text) is edited
    along with the region, so every such token must be known. One row per region; `match` is False
    where they differ."""
    regions = [("question", *prompt.spans["question"])] + [
        (s["role"], s["char_start"], s["char_end"]) for s in prompt.spans["sentences"]] + [
        ("instruction", a, b) for a, b in prompt.spans.get("instruction", [])]
    rows = []
    for cls, a, b in regions:
        toks = [(s, e) for i, (s, e) in enumerate(prompt.offsets)
                if prompt.classes[i] == cls and s < b and e > a]
        got = "".join(prompt.text[s:e] for s, e in toks)
        want = prompt.text[a:b]
        rows.append({
            "stimulus_id": prompt.stimulus_id, "question_key": prompt.question_key,
            "region": cls, "char_start": a, "char_end": b, "match": got == want,
            "n_tokens": len(toks),
            "first_token": prompt.text[slice(*toks[0])] if toks else None,
            "last_token": prompt.text[slice(*toks[-1])] if toks else None,
            "tokens_text": got, "region_text": want,
        })
    return rows


def mask(prompt: Prompt, classes: set[PositionClass], skip_first: int = 4) -> torch.Tensor:
    """Boolean mask over positions: True where prompt.classes[i] in `classes`, positions < skip_first
    forced False (the spec's hygiene invariant 3 skips the first ~4 high-norm positions; M0 used 4.
    From M2 on, skip_first is 0 -- the paper swaps across all question tokens, and the high-norm
    first positions are chat-template tokens, which no position set includes anyway)."""
    m = torch.tensor([c in classes for c in prompt.classes], dtype=torch.bool)
    if skip_first > 0:
        m[: min(skip_first, len(m))] = False
    return m


def mask_rows(prompt: Prompt, mask: torch.Tensor) -> list[dict]:
    """One row per token -- everything figures.mask_figure draws, as plain data for masks.parquet.
    `token_text` is the token's raw text sliced from `prompt.text` by its offsets ("" if the
    tokenizer gave it an empty span)."""
    rows = []
    for i, tid in enumerate(prompt.input_ids):
        s, e = prompt.offsets[i] if prompt.offsets else (0, 0)
        rows.append({
            "stimulus_id": prompt.stimulus_id,
            "question_key": prompt.question_key,
            "pos": i,
            "token_id": int(tid),
            "token_text": prompt.text[s:e],
            "class": prompt.classes[i],
            "edited": bool(mask[i]),
            "is_metric_pos": i == prompt.metric_pos,
        })
    return rows


def render_mask(prompt: Prompt, mask: torch.Tensor, path: Path) -> None:
    """Convenience wrapper: draw one prompt's mask via figures.mask_figure (the only drawing code)
    and save it to `path` (format from its suffix). Milestone scripts save mask_rows to parquet and
    call figures.make_figures instead, so the figure can be rebuilt later without the model."""
    import pandas as pd

    from . import figures

    figures.write(figures.mask_figure(pd.DataFrame(mask_rows(prompt, mask))), path)
