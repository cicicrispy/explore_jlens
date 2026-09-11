"""Cell/grid orchestration (M3).

ORCHESTRATION ASSUMPTIONS (flagged; the spec fixes the math and the outputs but not every wiring
detail between modules -- documented here rather than guessed silently):

- `cfgs` bundles runtime context not carried by `Cell` itself: `stimuli` (dict[id -> stimulus]),
  `fmt` (for prompts.build_prompt, see prompts.make_fmt), `tokens_raw` (raw configs/tokens.yaml),
  `tokens_cfg` (metrics.build_tokens_cfg output), `skip_first`, `save_topk`, `norm_scale`,
  `config_hash`, `lens_sha`, `model_revision`, `run_id`. `run_prompt` also injects `clean_cache`.
- Direction mapping: `pairs.<name>` is always ordered (es, fr). For a stimulus with matrix_lang and
  intrusion_lang in {es, fr}, "m2i" swaps FROM the matrix-language pair member TO the
  intrusion-language member at the intervened positions; "i2m" reverses it
  (controls.treatment_rows is the same mapping). The swap is symmetric in its two tokens, so the two
  directions' swap / big_nonlabel / random_direction cells are the same edit; only
  label_to_present differs (it removes a different label). Both are run (the literal grid).
- Control-token cells (label_to_present, big_nonlabel) carry their control in the Cell itself
  (`control_tokens`, picked automatically -- controls.py), several per kind (`control_index`).
- `run_prompt` runs one prompt's cells: `identity`, then `swap`, then the rest. `identity` populates
  `clean_cache` keyed by `_clean_key` (prompt only); `swap` populates `target_norms` keyed by
  `_norms_key` (prompt, direction, pair, layer set, alpha), which `big_nonlabel` /
  `random_direction` cells with the same key are scaled against.
- **Running a subset of kinds** (e.g. only `big_nonlabel` while iterating on `norm_scale`) is
  supported without re-running the cells it would normally depend on: pass pre-computed
  `target_norms_cache` and/or a pre-populated `cfgs["clean_cache"]`, built from a previous run's
  records via `target_norms_cache_from_records` / `clean_cache_from_records`. The official M3 runs
  always run all kinds together (hygiene invariant #7); this is for ad hoc use only.
"""
from __future__ import annotations

import dataclasses
import time
from collections import defaultdict
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import torch

from . import env as env_mod
from . import interventions as iv
from . import metrics
from . import prompts as prompts_mod


@dataclass(kw_only=True)
class Cell:
    stimulus_id: str
    question_key: str
    direction: str   # "m2i" | "i2m"
    pair_name: str
    kind: str        # "swap" | "label_to_present" | "big_nonlabel" | "random_direction" | "identity"
    layers: list
    position_set: set  # position classes edited, e.g. {"question"}
    alpha: float
    seed: int
    control_index: int = -1        # label_to_present / big_nonlabel: which of the run's controls (0, 1, 2)
    control_tokens: list = field(default_factory=list)  # token ids: [target] or [a, b]
    control_text: list = field(default_factory=list)    # their text, for the record
    control_tier: int = 0          # 1 (>=100%), 2 (>=75%), 3 (below 75%); 0 = not a control-token cell


@dataclass(kw_only=True)
class CellRecord(Cell):
    margin: float
    clean_margin: float          # NaN when no clean baseline was available
    flip: bool | None            # None when no clean baseline was available
    top1_changed: bool | None    # None when no clean baseline was available
    logprobs_fp16: np.ndarray
    topk: list                   # the model's top-`save_topk` next tokens at metric_pos
    intervention_logs: list
    prompt_len: int
    metric_pos: int
    git_commit: str
    config_hash: str
    lens_sha: str
    model_revision: str
    timestamp: float
    answer_label: str   # extra vs. the literal spec list: needed to compute `flip` for later cells
    top1_id: int         # extra vs. the literal spec list: needed to compute `top1_changed`
    s_token: str = ""    # the treatment pair's source / target words for this direction
    t_token: str = ""
    n_planned: int = 0            # planned (masked) positions
    n_planned_unchanged: int = 0  # (layer, position) entries planned but measured unchanged
    run_id: str = ""


def _resolve_pair_words(cfgs: dict, cell: Cell, matrix_lang: str) -> tuple[str, str]:
    es_word, fr_word = cfgs["tokens_raw"]["pairs"][cell.pair_name]
    matrix_word = es_word if matrix_lang == "es" else fr_word
    intrusion_word = fr_word if matrix_lang == "es" else es_word
    if cell.direction == "m2i":
        return matrix_word, intrusion_word
    if cell.direction == "i2m":
        return intrusion_word, matrix_word
    raise ValueError(f"unknown direction {cell.direction!r}")


def run_cell(model, lens, cell: Cell, cfgs: dict, target_norms: dict | None = None):
    """Run one cell. Returns (CellRecord, detail rows). The detail rows -- one per band layer x
    prompt position -- hold where the stream ACTUALLY changed (measured at every position, not only
    the planned ones) next to the planned intervention's log (hygiene invariant 4). A change at any
    position outside the plan raises RuntimeError: that is a hard failure."""
    stimulus = cfgs["stimuli"][cell.stimulus_id]
    matrix_lang = stimulus["matrix_lang"]

    prompt = prompts_mod.build_prompt(stimulus, cell.question_key, cfgs["fmt"])
    mask = prompts_mod.mask(prompt, set(cell.position_set), skip_first=cfgs.get("skip_first", 0))
    s_token, t_token = _resolve_pair_words(cfgs, cell, matrix_lang)

    if cell.kind == "swap":
        kw = dict(s_token=s_token, t_token=t_token, alpha=cell.alpha)
    elif cell.kind == "label_to_present":
        assert len(cell.control_tokens) == 1, "label_to_present needs exactly one control token"
        kw = dict(source_token=s_token, target_token=int(cell.control_tokens[0]), alpha=cell.alpha)
    elif cell.kind == "big_nonlabel":
        assert len(cell.control_tokens) == 2, "big_nonlabel needs a control token pair"
        assert target_norms is not None, "big_nonlabel requires target_norms from the treatment cell"
        kw = dict(a_token=int(cell.control_tokens[0]), b_token=int(cell.control_tokens[1]),
                  target_norms=target_norms, norm_scale=cfgs.get("norm_scale", 1.0))
    elif cell.kind == "random_direction":
        assert target_norms is not None, "random_direction requires target_norms from the treatment cell"
        kw = dict(target_norms=target_norms, seed=cell.seed)
    elif cell.kind == "identity":
        kw = {}
    else:
        raise ValueError(f"unknown kind {cell.kind!r}")

    logits, logs, changes = iv.apply(model, lens, prompt, cell.kind, cell.layers, mask, return_changes=True, **kw)
    outside, unchanged = iv.edit_problems(changes, mask, cell.kind)
    if outside:
        raise RuntimeError(f"{cell.stimulus_id}/{cell.question_key} {cell.direction} {cell.kind}"
                           f"[{cell.control_index}]: the stream changed at unplanned positions "
                           f"(layer, pos, size): {outside[:10]} -- hard failure, nothing saved for this prompt")

    logprobs = torch.log_softmax(logits, dim=-1)
    margin = metrics.question_margin(cell.question_key, logprobs, cfgs["tokens_cfg"], matrix_lang)
    answer_label = metrics.argmax_label(cell.question_key, logprobs, cfgs["tokens_cfg"], matrix_lang)
    top1_id = int(torch.argmax(logits))

    clean = cfgs.get("clean_cache", {}).get(_clean_key(cell))
    if cell.kind == "identity":
        clean_margin, flip, top1_changed = margin, False, False
    elif clean is None:
        # No clean baseline (subset run without identity or clean_cache): unknown, NOT "no effect".
        clean_margin, flip, top1_changed = float("nan"), None, None
    else:
        clean_margin = clean.margin
        flip = answer_label != clean.answer_label
        top1_changed = top1_id != clean.top1_id

    record = CellRecord(
        **{f.name: getattr(cell, f.name) for f in dataclasses.fields(Cell)},
        margin=margin,
        clean_margin=clean_margin,
        flip=flip,
        top1_changed=top1_changed,
        logprobs_fp16=logprobs.to(torch.float16).cpu().numpy(),
        topk=metrics.topk_tokens(logits, model.tokenizer, cfgs.get("save_topk", 100)),
        intervention_logs=logs,
        prompt_len=len(prompt.input_ids),
        metric_pos=prompt.metric_pos,
        git_commit=env_mod.git_commit(),
        config_hash=cfgs.get("config_hash", ""),
        lens_sha=cfgs.get("lens_sha", ""),
        model_revision=cfgs.get("model_revision", ""),
        timestamp=time.time(),
        answer_label=answer_label,
        top1_id=top1_id,
        s_token=s_token,
        t_token=t_token,
        n_planned=int(mask.sum()),
        n_planned_unchanged=len(unchanged),
        run_id=cfgs.get("run_id", ""),
    )
    return record, _detail_rows(cell, prompt, mask, logs, changes)


def _detail_rows(cell: Cell, prompt, mask, logs, changes) -> list[dict]:
    """One row per (layer, position): the measured change next to the planned intervention's log."""
    by_lp = {(lg.layer, lg.pos): lg for lg in logs}
    planned = mask.tolist()
    nan = float("nan")
    rows = []
    for layer, sizes in sorted(changes.items()):
        for pos, size in enumerate(sizes):
            lg = by_lp.get((layer, pos))
            rows.append({
                "stimulus_id": cell.stimulus_id, "question_key": cell.question_key,
                "direction": cell.direction, "kind": cell.kind, "control_index": cell.control_index,
                "layer": int(layer), "pos": pos, "class": prompt.classes[pos], "planned": bool(planned[pos]),
                "change": float(size),
                "c_before_0": lg.c_before[0] if lg else nan, "c_before_1": lg.c_before[1] if lg else nan,
                "c_after_0": lg.c_after[0] if lg else nan, "c_after_1": lg.c_after[1] if lg else nan,
                "delta_c_norm": lg.delta_c_norm if lg else nan, "delta_h_norm": lg.delta_h_norm if lg else nan,
                "alpha": lg.alpha if lg else nan,
            })
    return rows


def _clean_key(rd) -> tuple:
    """Key for the clean baseline. Identity output depends only on the prompt, so direction, pair,
    layers and alpha are deliberately NOT part of it."""
    g = rd.get if isinstance(rd, dict) else lambda k: getattr(rd, k)
    return (g("stimulus_id"), g("question_key"))


def _norms_key(rd) -> tuple:
    """Key matching a control to its treatment (swap) cell: same prompt, direction, pair, layer set
    and alpha. For control cells, `alpha` means "the treatment alpha this control is matched to"."""
    g = rd.get if isinstance(rd, dict) else lambda k: getattr(rd, k)
    return (g("stimulus_id"), g("question_key"), g("direction"), g("pair_name"),
            tuple(sorted(int(l) for l in g("layers"))), float(g("alpha")))


def _target_norms_from_logs(record: CellRecord) -> dict:
    by_layer_pos = {}
    for log in record.intervention_logs:
        by_layer_pos[(log.layer, log.pos)] = log.delta_h_norm
    norms = {}
    for l in record.layers:
        norms[l] = torch.tensor(
            [by_layer_pos.get((l, p), 0.0) for p in range(record.prompt_len)]
        )
    return norms


def target_norms_cache_from_records(records) -> dict:
    """Build a `target_norms_cache` (see `run_prompt`) from a previous run's `swap` records -- either
    live `CellRecord`s or rows read back from parquet (plain dicts; `intervention_logs` then comes
    back as a list of dicts too, since the parquet writers flatten dataclasses for storage). Lets
    `big_nonlabel`/`random_direction` be run later, standalone, without re-running `swap`."""
    cache = {}
    for r in records:
        rd = r if isinstance(r, dict) else dataclasses.asdict(r)
        if rd["kind"] != "swap":
            continue
        by_layer_pos = {}
        for log in rd["intervention_logs"]:
            lg = log if isinstance(log, dict) else dataclasses.asdict(log)
            by_layer_pos[(lg["layer"], lg["pos"])] = lg["delta_h_norm"]
        cache[_norms_key(rd)] = {
            int(l): torch.tensor([by_layer_pos.get((l, p), 0.0) for p in range(rd["prompt_len"])])
            for l in rd["layers"]
        }
    return cache


def clean_cache_from_records(records) -> dict:
    """Build a `cfgs["clean_cache"]` from a previous run's `identity` records (live or from
    parquet, see `target_norms_cache_from_records`). Lets `flip`/`top1_changed` be computed against
    an existing clean pass without re-running `identity`."""
    cache = {}
    for r in records:
        rd = r if isinstance(r, dict) else dataclasses.asdict(r)
        if rd["kind"] != "identity":
            continue
        cache[_clean_key(rd)] = SimpleNamespace(
            margin=rd["margin"], answer_label=rd["answer_label"], top1_id=rd["top1_id"]
        )
    return cache


KIND_ORDER = {"identity": 0, "swap": 1, "label_to_present": 2, "big_nonlabel": 3, "random_direction": 4}


def run_prompt(model, lens, cells: list, cfgs: dict, target_norms_cache: dict | None = None):
    """Run every cell of ONE prompt, `identity` first, then `swap`, then the controls (see the
    module docstring). Returns (records, detail rows). `cfgs["clean_cache"]` and
    `target_norms_cache` are updated in place so later prompts/cells can reuse them."""
    keys = {_clean_key(c) for c in cells}
    assert len(keys) == 1, f"run_prompt expects the cells of one prompt, got {sorted(keys)}"
    cfgs.setdefault("clean_cache", {})
    target_norms_cache = {} if target_norms_cache is None else target_norms_cache
    records, details = [], []
    for c in sorted(cells, key=lambda c: (KIND_ORDER.get(c.kind, 99), c.direction, c.control_index)):
        tn = target_norms_cache.get(_norms_key(c)) if c.kind in ("big_nonlabel", "random_direction") else None
        record, rows = run_cell(model, lens, c, cfgs, target_norms=tn)
        if c.kind == "identity":
            cfgs["clean_cache"][_clean_key(c)] = record
        if c.kind == "swap":
            target_norms_cache[_norms_key(c)] = _target_norms_from_logs(record)
        records.append(record)
        details += rows
    return records, details


def group_by_prompt(cells: list) -> dict:
    """{(stimulus_id, question_key): [cells]}, in the order prompts first appear."""
    groups = defaultdict(list)
    for c in cells:
        groups[_clean_key(c)].append(c)
    return dict(groups)
