"""Cell/grid orchestration.

ORCHESTRATION ASSUMPTIONS (flagged; the spec fixes the math and the outputs but not every wiring
detail between modules -- documented here rather than guessed silently):

- `cfgs` bundles runtime context not carried by `Cell` itself: `stimuli` (dict[id -> stimulus]),
  `fmt` (for prompts.build_prompt: prompt_format.yaml fields + "tokenizer" + "questions"),
  `tokens_raw` (raw configs/tokens.yaml), `tokens_cfg` (metrics.build_tokens_cfg output),
  `skip_first`, `config_hash`, `lens_sha`, `model_revision`. `run_grid` also injects `clean_cache`.
- Direction mapping: `pairs.<name>` is always ordered (es, fr). For a stimulus with matrix_lang and
  intrusion_lang in {es, fr}, "m2i" swaps FROM the matrix-language pair member TO the
  intrusion-language member at the (question-only) intervened positions; "i2m" reverses it.
- `run_grid` groups cells by prompt and runs `identity`, then `swap`, then the rest. `identity`
  populates `clean_cache` keyed by `_clean_key` (prompt only); `swap` populates `target_norms` keyed
  by `_norms_key` (prompt, direction, pair, layer set, alpha), which `big_nonlabel` /
  `random_direction` cells with the same key are scaled against.
- **Running a subset of kinds** (e.g. only `big_nonlabel` while iterating on `norm_scale`, or
  skipping `identity` because you don't care about `flip` yet) is supported without re-running the
  cells it would normally depend on: pass pre-computed `target_norms_cache` (keyed the same way as
  the internal grouping) and/or a pre-populated `cfgs["clean_cache"]`, built from a previous run's
  records via `target_norms_cache_from_records` / `clean_cache_from_records` below. `scripts/
  m3_grid.py`'s official M3 run still runs all 5 kinds together every time, per the spec's hygiene
  invariant #7 (no treatment cell without its controls in the same run) -- this flexibility is for
  ad hoc/exploratory use, not a change to what M3 itself runs.
"""
from __future__ import annotations

import dataclasses
import time
from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch

from . import env as env_mod
from . import interventions as iv
from . import io as io_mod
from . import metrics
from . import prompts as prompts_mod


@dataclass
class Cell:
    stimulus_id: str
    question_key: str
    direction: str   # "m2i" | "i2m"
    pair_name: str
    kind: str        # "swap" | "label_to_present" | "big_nonlabel" | "random_direction" | "identity"
    layers: list
    position_set: set
    alpha: float
    seed: int


@dataclass
class CellRecord(Cell):
    margin: float
    clean_margin: float          # NaN when no clean baseline was available
    flip: bool | None            # None when no clean baseline was available
    top1_changed: bool | None    # None when no clean baseline was available
    logprobs_fp16: np.ndarray
    top5: list
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


def _resolve_pair_words(cfgs: dict, cell: Cell, matrix_lang: str) -> tuple[str, str]:
    es_word, fr_word = cfgs["tokens_raw"]["pairs"][cell.pair_name]
    matrix_word = es_word if matrix_lang == "es" else fr_word
    intrusion_word = fr_word if matrix_lang == "es" else es_word
    if cell.direction == "m2i":
        return matrix_word, intrusion_word
    if cell.direction == "i2m":
        return intrusion_word, matrix_word
    raise ValueError(f"unknown direction {cell.direction!r}")


def run_cell(model, lens, cell: Cell, cfgs: dict, target_norms: dict | None = None) -> CellRecord:
    stimulus = cfgs["stimuli"][cell.stimulus_id]
    matrix_lang = stimulus["matrix_lang"]

    prompt = prompts_mod.build_prompt(stimulus, cell.question_key, cfgs["fmt"])
    mask = prompts_mod.mask(prompt, cell.position_set, skip_first=cfgs.get("skip_first", 4))
    s_token, t_token = _resolve_pair_words(cfgs, cell, matrix_lang)

    controls = cfgs["tokens_raw"]["controls"]
    if cell.kind == "swap":
        kw = dict(s_token=s_token, t_token=t_token, alpha=cell.alpha)
    elif cell.kind == "label_to_present":
        target = controls["label_to_present"]["target"]
        assert target is not None, "controls.label_to_present.target is null; M3 must not run"
        kw = dict(source_token=s_token, target_token=target, alpha=cell.alpha)
    elif cell.kind == "big_nonlabel":
        pair_bn = controls["big_nonlabel"]["pair"]
        assert pair_bn is not None, "controls.big_nonlabel.pair is null; M3 must not run"
        assert target_norms is not None, "big_nonlabel requires target_norms from the treatment cell"
        kw = dict(
            a_token=pair_bn[0],
            b_token=pair_bn[1],
            target_norms=target_norms,
            norm_scale=controls["big_nonlabel"].get("norm_scale", 1.0),
        )
    elif cell.kind == "random_direction":
        assert target_norms is not None, "random_direction requires target_norms from the treatment cell"
        kw = dict(target_norms=target_norms, seed=cell.seed)
    elif cell.kind == "identity":
        kw = {}
    else:
        raise ValueError(f"unknown kind {cell.kind!r}")

    logits, logs = iv.apply(model, lens, prompt, cell.kind, cell.layers, mask, **kw)
    logprobs = torch.log_softmax(logits, dim=-1)
    margin = metrics.question_margin(cell.question_key, logprobs, cfgs["tokens_cfg"], matrix_lang)
    answer_label = metrics.argmax_label(cell.question_key, logprobs, cfgs["tokens_cfg"], matrix_lang)
    top1_id = int(torch.argmax(logits))

    clean_key = _clean_key(cell)
    clean = cfgs.get("clean_cache", {}).get(clean_key)
    if cell.kind == "identity":
        clean_margin, flip, top1_changed = margin, False, False
    elif clean is None:
        # No clean baseline (subset run without identity or clean_cache): unknown, NOT "no effect".
        clean_margin, flip, top1_changed = float("nan"), None, None
    else:
        clean_margin = clean.margin
        flip = answer_label != clean.answer_label
        top1_changed = top1_id != clean.top1_id

    tokenizer = model.tokenizer
    top5_vals, top5_ids = torch.topk(logits, 5)
    top5 = [{"token": tokenizer.decode([int(i)]), "logit": float(v)} for v, i in zip(top5_vals, top5_ids)]

    return CellRecord(
        stimulus_id=cell.stimulus_id,
        question_key=cell.question_key,
        direction=cell.direction,
        pair_name=cell.pair_name,
        kind=cell.kind,
        layers=cell.layers,
        position_set=cell.position_set,
        alpha=cell.alpha,
        seed=cell.seed,
        margin=margin,
        clean_margin=clean_margin,
        flip=flip,
        top1_changed=top1_changed,
        logprobs_fp16=logprobs.to(torch.float16).cpu().numpy(),
        top5=top5,
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
    )


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
    """Build a `target_norms_cache` (see `run_grid`) from a previous run's `swap` records -- either
    live `CellRecord`s or rows read back from parquet (plain dicts; `intervention_logs` then comes
    back as a list of dicts too, since `io.append_records` flattens dataclasses for storage). Lets
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


def run_grid(model, lens, cells: list, cfgs: dict, out_path, uploader=None, target_norms_cache: dict | None = None):
    """Yields CellRecords, writing each to `out_path` (parquet) as it's produced.

    Cells are grouped by prompt (stimulus_id, question_key); within a prompt, `identity` cells run
    first, then `swap`, then the rest (by convention, not requirement). `identity` feeds
    `cfgs["clean_cache"]`, keyed by prompt only (a clean pass doesn't depend on direction, pair,
    layers or alpha). `swap` feeds `target_norms`, keyed by `_norms_key` (prompt, direction, pair,
    layer set, alpha), so two swaps with different bands or alphas in one call can never hand their
    norms to each other's controls.

    To run a subset of kinds without the cells they'd otherwise depend on: pass a pre-populated
    `cfgs["clean_cache"]` (skip needing `identity`) and/or `target_norms_cache` (skip needing
    `swap`), both buildable from a previous run's records via `clean_cache_from_records` /
    `target_norms_cache_from_records`. A group that needs `target_norms` (has a `big_nonlabel` or
    `random_direction` cell) and has neither a `swap` cell in `cells` nor an entry in
    `target_norms_cache` still raises clearly in `run_cell` -- scaling those controls against
    nothing would be meaningless, not just unspecified.

    `uploader`, if given an `io.BackgroundUploader`, has `.trigger()` called after every record is
    appended -- this runs `hf upload` on a background thread (at most one at a time; see
    `io.BackgroundUploader`) so syncing the growing parquet off the GPU box never stalls the trace
    loop here. The caller is responsible for constructing the uploader and calling `.flush()` after
    this generator is exhausted, to block until the final upload has actually landed."""
    groups = defaultdict(list)
    for c in cells:
        groups[_clean_key(c)].append(c)

    kind_order = {"identity": 0, "swap": 1, "label_to_present": 2, "big_nonlabel": 3, "random_direction": 4}
    cfgs = dict(cfgs)
    cfgs["clean_cache"] = dict(cfgs.get("clean_cache", {}))
    target_norms_cache = dict(target_norms_cache or {})

    for key, group_cells in groups.items():
        group_cells = sorted(group_cells, key=lambda c: kind_order.get(c.kind, 99))
        for c in group_cells:
            tn = target_norms_cache.get(_norms_key(c)) if c.kind in ("big_nonlabel", "random_direction") else None
            record = run_cell(model, lens, c, cfgs, target_norms=tn)
            if c.kind == "identity":
                cfgs["clean_cache"][key] = record
            if c.kind == "swap":
                target_norms_cache[_norms_key(c)] = _target_norms_from_logs(record)
            io_mod.append_records([record], out_path)
            if uploader is not None:
                uploader.trigger()
            yield record
