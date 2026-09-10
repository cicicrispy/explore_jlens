"""Prompt construction and per-token position classification.

CALL CONVENTION NOTE (flagged assumption, not literally specified): `build_prompt`'s contract is
`(stimulus, question_key, fmt)` with no explicit tokenizer/questions-map argument. Since both are
required to build anything, this implementation expects the caller to fold them into `fmt`:
    fmt = {**yaml_loaded_prompt_format, "tokenizer": tokenizer, "questions": stimuli_json["questions"]}
This is the only way to satisfy the 3-argument signature; flagged here for the human to confirm or
correct rather than silently inventing a 4th parameter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

PositionClass = Literal["template", "question", "matrix", "intrusion"]

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


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def build_prompt(stimulus: dict, question_key: str, fmt: dict) -> Prompt:
    tokenizer = fmt["tokenizer"]
    question_text = fmt["questions"][question_key]
    passage_text = stimulus["text"]

    messages = [{"role": "user", "content": f"{question_text}\n\n{passage_text}"}]

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

    q_start = templated.find(question_text)
    assert q_start != -1, "question text not found verbatim in templated prompt"
    q_end = q_start + len(question_text)

    p_start = templated.find(passage_text, q_end)
    assert p_start != -1, "passage text not found verbatim in templated prompt after the question"

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
        overlaps = {"question": q_ov, "matrix": m_ov, "intrusion": i_ov}
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
        "passage": [p_start, p_start + len(passage_text)],
        "sentences": sentence_spans,
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
    )


def mask(prompt: Prompt, classes: set[PositionClass], skip_first: int = 4) -> torch.Tensor:
    """Boolean mask over positions: True where prompt.classes[i] in `classes`, positions < skip_first
    forced False (skips the first ~4 high-norm positions per the hygiene invariant)."""
    m = torch.tensor([c in classes for c in prompt.classes], dtype=torch.bool)
    if skip_first > 0:
        m[: min(skip_first, len(m))] = False
    return m


_CLASS_COLORS = {
    "template": "#cccccc",
    "question": "#7fb3ff",
    "matrix": "#7fdc7f",
    "intrusion": "#ffb37f",
}


def render_mask(prompt: Prompt, mask: torch.Tensor, path: Path) -> None:
    """PNG: one box per token, colored by class, outlined where mask is True, metric_pos marked."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n = len(prompt.input_ids)
    cols = 16
    rows = (n + cols - 1) // cols
    fig, ax = plt.subplots(figsize=(cols * 1.1, rows * 0.9))

    for i in range(n):
        row = i // cols
        col = i % cols
        x, y = col, rows - 1 - row
        color = _CLASS_COLORS.get(prompt.classes[i], "#ffffff")
        edge = "black"
        lw = 1.0
        if bool(mask[i]):
            edge = "red"
            lw = 2.5
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color, edgecolor=edge, linewidth=lw))
        if i == prompt.metric_pos:
            ax.plot(x + 0.5, y + 0.5, marker="*", color="black", markersize=14)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{prompt.stimulus_id} / {prompt.question_key}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
