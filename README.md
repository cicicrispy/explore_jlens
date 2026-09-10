# explore_jlens -- Stage 0/1 replication

J-lens selectivity replication. See the coding prompt in this repo's history for the full spec;
this README covers how to run each milestone.

## Status

Code has been written for M0 (Phase A, Mac) and M1-M3 (Phase B, GPU box), but **nothing has been
executed yet**. Commands below are what to run; none of them have been run by the assistant that
wrote this code -- run them yourself and report back what happens (errors, test failures, etc.) so
we can iterate. See "Known open items" below for things flagged during writing that need your input
before M1 can run for real.

## Setup

```bash
bash setup.sh
```

Idempotent; detects mac/cuda/cpu-linux, installs torch appropriately, installs the package
editable, and (on `cuda` only) downloads the model + lens and runs the GPU-marked tests too.
Requires `.env` (copy `.env.example`, fill in `HF_TOKEN`; `GH_TOKEN` is optional -- only needed if
this machine doesn't already have git/GitHub access configured).

**Every new shell, before running any script:** `source .venv/bin/activate`.

**Never `source .env` in a shell.** Secrets are loaded only from Python: every script (and the
test suite, and `setup.sh`'s secret check) calls `env.bootstrap()` first, which reads `.env` with
`python-dotenv` into that process's environment -- values are never printed, and variables already
set are not overridden. `bootstrap()` also changes to the repo root (so all paths are repo-relative
and scripts work from any directory) and sets `HF_HOME` from `configs/paths.yaml`, so a fresh shell
reuses the cache `setup.sh` filled instead of re-downloading.

**No code ever writes to `configs/`.** Values discovered at runtime (resolved lens sha, observed
chat template, single-token checks, proposed controls) are written under `runs/<M>/`; you copy
anything that should become canonical into `configs/` by hand.

**Commit before running a milestone.** Records store the git commit; if code or configs differ
from it, the hash is suffixed `-dirty`.

## M0 (Phase A, Mac -- run this first)

```bash
pytest tests/ -q
python scripts/m0_smoke.py
```

Builds all 64 stimulus x question prompts on the stand-in model (`configs/model.yaml`'s
`standin_hf_id`, `Qwen/Qwen3-0.6B`), renders mask PNGs to
`runs/M0/figures/masks/`, runs one `swap` and one `identity` cell with a random lens, and writes
`runs/M0/summary.md`, `runs/M0/template_string.txt` and `runs/M0/single_token_check.yaml`. Uploads `runs/M0/` to the `orbitsoferis/jlens-specificity` HF dataset at the
end (needs `HF_TOKEN`).

## M1 (Phase B, GPU box -- only after M0 sign-off)

```bash
bash setup.sh                 # on the GPU box; runs scripts/download.py
# copy revision_sha from runs/M1/lens_resolved.yaml into configs/lens.yaml
python scripts/m1_validate.py
```

`scripts/download.py` fetches `Qwen/Qwen3.6-27B` and the lens, and writes the resolved lens sha,
covered layers, stored dtype, coverage ratio and `CREDIT.md` to `runs/M1/`. The model is a
vision-language checkpoint; `model.decoder()` locates its text decoder, and only text is fed in.
The antonym positive-control prompt in `scripts/m1_validate.py` (`ANTONYM_USER_TEXT`) is a
placeholder until replaced with the paper's exact prompt.

Runs lens validation: final-layer agreement, readout reproduction on `sp_01`, the Chinese-antonym
causal positive control (stops before M2 if it fails at both alpha=1 and alpha=2), and band
signatures (CKA + next-token agreement + kurtosis by layer) into
`runs/M1/band_signatures.{parquet,png}`. **You** then fill `configs/bands.yaml`'s `workspace` (and
`full`/`early_late`, used later) by reading that figure -- no code does this automatically.

## M2 (Phase B -- only after `configs/bands.yaml.workspace` is filled)

```bash
python scripts/m2_loading.py
```

Clean-pass workspace loading over all 64 prompts, an accuracy table, and rule-based control-token
proposals written to `runs/M2/controls_proposed.yaml`. Review that file, optionally extend
`controls.control_ineligible` in `configs/tokens.yaml` and re-run (cheap, no forward passes needed
for re-selection -- currently re-running the whole script is the only wired path; let me know if
you want a standalone re-selection script instead), then copy the approved
`label_to_present_target`/`big_nonlabel_pair` into `configs/tokens.yaml`'s `controls` block.

## M3 (Phase B -- only after M2's controls are approved)

```bash
python scripts/m3_grid.py
```

Refuses to run if `bands.workspace`, `lens.revision_sha`, `controls.label_to_present.target`, or
`controls.big_nonlabel.pair` is null. Runs the full grid (16 stimuli x 4 questions x 2 directions x
1 pair x 5 kinds), writes `runs/M3/records.parquet`, `panel_c.png`, `margin_vs_deltac_*.png`, and
ten sampled raw cells per group into `summary.md`.

## Layer bands: contiguous ranges vs. multiple blocks

**Every `[a, b]` in `configs/bands.yaml` is inclusive on both ends**: `[18, 30]` means layers 18
through 30. `workspace` (used by M2/M3) is one pair, `[onset, motor_onset - 1]`; `full` and
`early_late` (Stage 2) may be a pair or a list of pairs, e.g. `[[18, 30], [40, 50]]`.
`env.expand_band(band_spec)` flattens either shape into one sorted, deduplicated `list[int]`.
`interventions.apply` refuses any layer the lens doesn't cover, which includes the final layer.

That flat list is all `interventions.apply` and `runner.py` ever see -- they loop over `layers` in
ascending order inside a single trace and don't care whether those layers are one contiguous run or
several disjoint blocks glued together. So a "swap across layers 18-30 and 40-50" intervention is
just `apply(..., layers=env.expand_band([[18, 30], [40, 50]]), ...)`: one cell, one trace, every
layer in both blocks edited together, each seeing the previously-edited stream from earlier layers
in the list (the "clamped" behavior). `scripts/m2_loading.py` and `scripts/m3_grid.py` already build
their layer lists via `env.expand_band(bands["workspace"])`.

## Background uploads

M2's clean pass and M3's grid can run long enough on the GPU box that waiting until the very end to
run `hf upload` would risk losing a lot of work to a crash, and blocking on it periodically would
stall the trace loop for no reason (GPU work doesn't need the network). `io.BackgroundUploader`
runs `upload_run` on a single background thread instead:

```python
uploader = io.BackgroundUploader(run_dir)
...
io.append_records([record], out_path)
uploader.trigger()          # cheap; returns immediately, never blocks the loop
...
uploader.flush()            # blocks until the last triggered upload has actually landed
uploader.shutdown()
```

At most one `hf upload` subprocess runs at a time. Calling `trigger()` while one is already in
flight doesn't start a second (concurrent uploads to the same repo path would race) -- it just marks
the uploader "dirty" so exactly one more upload runs immediately after the current one finishes,
picking up everything written since. `runner.run_grid` takes an optional `uploader=` argument and
calls `.trigger()` itself after every record; `scripts/m2_loading.py` and `scripts/m3_grid.py` both
wire this up already, then `flush()` + `shutdown()` before doing one final, synchronous `upload_run`
call at the very end so the run's figures and `summary.md` (written after the loop) are included
too. Upload failures (e.g. `HF_TOKEN` unset, rate limits) never raise into the GPU loop, but each
one is printed to stderr as it happens, counted (`n_failures`), and reported in the milestone's
`summary.md`. `BackgroundUploader(run_dir, min_interval_s=...)` optionally spaces uploads apart
(default 0, no delay; any wait is on the upload thread, never the GPU loop). Parquet appends are
atomic (temp file + rename), so an upload never ships a half-written file. Upload paths are
repo-relative (`runs/<M>`), matching the layout inside the dataset.

## Running a subset of cells (tinkering, not the official M3 grid)

`scripts/m3_grid.py` always builds all 5 kinds together for every stimulus/question/direction, per
the spec's hygiene invariant #7 (no treatment cell without its controls in the same run) -- that
doesn't change. But `runner.run_grid`/`run_cell` don't actually require all 5 kinds to be present in
a given call; the only hard requirement is that `big_nonlabel`/`random_direction` need a
`target_norms` from somewhere, and `flip`/`top1_changed` need a clean baseline from somewhere. Both
can be supplied without re-running `swap`/`identity`:

```python
import yaml
from jlens_spec import io, runner

# from a previous run's records.parquet (or a list of freshly produced CellRecords)
old = pd.read_parquet("runs/M3/records.parquet").to_dict("records")

target_norms_cache = runner.target_norms_cache_from_records(old)   # from the old `swap` rows
cfgs["clean_cache"] = runner.clean_cache_from_records(old)          # from the old `identity` rows

# now iterate on just big_nonlabel, e.g. trying a different norm_scale, without re-running swap:
cells = [runner.Cell(..., kind="big_nonlabel", ...) for ...]
records = list(runner.run_grid(model, lens, cells, cfgs, "runs/scratch/records.parquet",
                                target_norms_cache=target_norms_cache))
```

Keys: the clean baseline is keyed by prompt only (`stimulus_id, question_key`), since an identity
pass doesn't depend on direction, pair, layers or alpha. Target norms are keyed by prompt,
direction, pair, layer set and alpha, so a control is only ever scaled against the swap it matches
(for control cells, `alpha` means the treatment alpha they're matched to). A `big_nonlabel` /
`random_direction` cell with no matching swap in the batch or cache raises clearly. A cell with no
clean baseline records `flip=None`, `top1_changed=None`, `clean_margin=NaN` -- unknown, not "no
effect" -- and `panel_c` leaves those out of the flip rates.

## Known open items (flagged during writing, need your input)

- **Antonym prompt** (`ANTONYM_USER_TEXT` in `scripts/m1_validate.py`) is a placeholder; replace it
  with the paper's exact prompt if available.
- **Lens file layout** is parsed heuristically (`lens.load_lens`); `runs/M1/lens_resolved.yaml`
  shows what was found. Also confirm lens keys are block indices (matching `decoder.layers[l]`),
  not hidden_states indices -- an off-by-one there would be silent.
- **`content_probe` -> `content`**: the spec's module contracts (`metrics.question_margin`, M2's
  accuracy table) mention `content_probe`, but `stimuli/stimuli.json`'s `questions` key is
  `content` (per `stimuli/notes.md`'s revision log). All code here uses `content` -- confirmed with
  you already, noting it here for anyone else reading this repo.
- **`prompts.build_prompt(stimulus, question_key, fmt)`** needs a tokenizer and the
  question-text mapping, neither of which fit the literal 3-argument contract. This implementation
  expects callers to fold them into `fmt`: `fmt = {**prompt_format_yaml, "tokenizer": tok,
  "questions": stimuli_json["questions"]}`. Flagged in `prompts.py`'s module docstring.
- Several other module-boundary assumptions (nnsight envoy call semantics outside a trace, the
  `.pt` lens artifact's exact key layout, `run_grid`'s cell-ordering/grouping logic) are documented
  inline where they occur (search for "ASSUMPTION" / "flagged" in `src/jlens_spec/`). None of this
  has touched a real nnsight trace or the real lens file yet, so treat first-run errors there as
  expected, not a sign anything is fundamentally wrong.
- `test_interventions.py`'s check 7 ("apply reproduces the mini-paper's `run` to 1e-4") is skipped
  -- no reference implementation is available in this repo to compare against.

## Secrets

`.env` (not tracked by git) holds `HF_TOKEN` and optionally `GH_TOKEN`. Nothing in this codebase
ever prints, logs, or hardcodes either value.
