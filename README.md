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
chat template, single-token checks, proposed controls) are written into the run's folder under
`runs/`; you copy anything that should become canonical into `configs/` by hand.

## Run folders and settings copies

Every run gets its own folder, and nothing is ever overwritten:

```
runs/<milestone>/<experiment name>_<UTC start time>/     e.g. runs/M0/smoke_20260912-031000/
    settings/        copies of every settings file the run uses, taken when the run starts
    manifest.json    fixed facts about the run, written once at the start and never changed:
                     run_id, milestone, experiment file, settings sources, config_hash, git
                     commit, start time
    figures/png/     figures drawn at the end of the run (other formats: see "Figures" below)
    ...              the run's data files (parquet etc.)
    summary.md       the milestone summary -- written LAST, see "When is a run finished?"
```

**When is a run finished? When its `summary.md` is on HF -- nothing else counts.** At the end of
every milestone script (`io.finalize_run`):

1. the run folder is uploaded to the HF dataset. `summary.md` doesn't exist yet, so it can't go up
   early. (`hf upload` may split a larger folder into several commits; that's fine.)
2. only after that upload has fully succeeded, `summary.md` is written -- including the run's HF
   folder link and the upload's URL -- and uploaded on its own, as the last step.

If step 1 fails, no summary is written and the error is saved to `upload_error.txt`. If step 2
fails, the summary is renamed `summary_not_uploaded.md` and the error is saved to
`upload_error.txt`. Either way, no `summary.md` means the run is not finished -- on HF and on your
machine. A run stopped mid-upload (e.g. Ctrl-C) also has no `summary.md`.

**Experiment files.** What a run does is described by a hand-written experiment file in
`configs/experiments/` (e.g. `configs/experiments/m0_smoke.yaml`), passed with `--experiment`.

**Settings copy.** When a run starts, the experiment file and every other settings file the run
reads (e.g. `configs/model.yaml`, `configs/tokens.yaml`, `stimuli/stimuli.json`) are copied into
that run's `settings/` folder. The run then reads its settings **only from that copy**, never from
`configs/`. Editing `configs/` while a run is going, or before resuming it, never changes that run;
the edit applies to the next new run. Code is not copied: every saved row records the git commit
that produced it. `config_hash` in the manifest and in the rows is a hash of the `settings/` folder.

Every saved row also carries the run's `run_id`. **Currently M0 and M1 use run folders**; M2 and
M3 still write to `runs/<M>/` until they are converted (next stage of work).

**Commit before running a milestone.** Records store the git commit; if code or configs differ
from it, the hash is suffixed `-dirty`.

## M0 (Phase A, Mac -- run this first)

```bash
pytest tests/ -q
python scripts/m0_smoke.py
```

Settings come from `configs/experiments/m0_smoke.yaml` (or `--experiment <file>`). Creates a new
run folder `runs/M0/smoke_<UTC start time>/` and, inside it:

- builds all 64 stimulus x question prompts on the stand-in model (`configs/model.yaml`'s
  `standin_hf_id`, `Qwen/Qwen3-0.6B`) and saves every prompt's per-token mask data (token text,
  class, edited or not, metric position) to `masks.parquet`;
- draws the 64 mask figures **from `masks.parquet`** into `figures/png/masks/`. A token with no text
  is labelled `#<token id>` (explained in each figure's legend);
- runs one `swap` and one `identity` cell with a random lens and saves them, with each cell's
  top-100 next tokens, to `m0_smoke.parquet`;
- writes `template_string.txt` and `single_token_check.yaml` (and `manifest.json` at the start).

At the end the run folder is uploaded to the `orbitsoferis/jlens-specificity` HF dataset (needs
`HF_TOKEN`) -- **without `figures/`**, since every M0 figure can be redrawn from `masks.parquet` --
and then `summary.md` is written and uploaded last (see "When is a run finished?" above).

## M1 (lens validation -- only after M0 sign-off)

M1 has three kinds of run, all under `runs/M1/`, each in its own folder and uploaded to HF:

| Step | Where | Command | Run folder |
|---|---|---|---|
| 1. token checks | **your Mac** (tokenizer only, no weights) | `python scripts/m1_tokens.py` | `runs/M1/tokens_<time>/` |
| 2. download | GPU box (`setup.sh` runs it) | `python scripts/download.py` | `runs/M1/download_<model>_<time>/` |
| 3. validation | GPU box | `python scripts/m1_validate.py` | `runs/M1/validate_<time>/` |

**1. Token checks (`configs/experiments/m1_tokens.yaml`).** Loads only the real model's tokenizer and
chat template and records: the region check for all 64 prompts (`region_check.parquet` -- the check
the stand-in test runs, see "Known open items"), the real template string and its diff against the
M0 run named in the experiment file, the single-token table for `configs/tokens.yaml` (spec M1 step 5),
and how the positive-control prompt splits into tokens. Run it before renting the GPU, so prompts or
stimuli can be adjusted first.

**2. Download (`configs/experiments/m1_download.yaml`).** Fetches `Qwen/Qwen3.6-27B` (into `HF_HOME`,
not loaded) and the lens, resolves the lens repo's current version, and records the version, covered
layers, stored dtype, one matrix's shape, coverage ratio and `CREDIT.md` in the run folder
(`lens_resolved.yaml`). Stops if the lens size doesn't match the model's hidden size. Then, by hand:
copy `revision_sha` into `configs/lens.yaml`, and put the run's folder name into
`configs/experiments/m1_validate.yaml`'s `download_run`. The model is a vision-language checkpoint;
`model.decoder()` locates its text decoder, and only text is fed in.

**3. Validation (`configs/experiments/m1_validate.yaml`).** Spec M1 steps 1-4, each saving the top-100
tokens (check 4: the rank of the model's real next token) so the k each check is judged at is applied
afterwards from the saved files (`src/jlens_spec/m1_checks.py`):
- final-layer agreement (lens readout vs the model's actual logits);
- readout reproduction on `sp_01` (figure `readout_top1_sp01`);
- **the Chinese-antonym causal positive control, exactly as in the paper**: the raw prompt
  `"小"的反义词是"` (no chat template), ` big`->` long` and ` bigger`->` longer` swapped at **every** token
  position across layers 25-75% of depth; 长 should become the top-1 answer instead of 大. Alpha 2 runs
  only if alpha 1 fails; if both fail, M1 stops (do not run M2). Where the stream actually changed is
  measured at every position, and any change outside the planned positions stops the run; the mask
  figures (`masks/check3_*`) are drawn from those measured changes.
- band signatures (CKA + next-token agreement + kurtosis by layer) into `band_signatures.parquet` and
  `figures/png/band_signatures.png`. **You** then fill `configs/bands.yaml`'s `workspace` (and
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

## Figures (regenerable without the model)

Compute and plotting are separate. Every milestone script saves the data behind each figure to its
run folder (parquet, plus a small `figure_params.json` for the few non-tabular values such as the
band used), and then builds its figures **from those files** via `figures.make_figures` -- never from
in-memory results. The same function is available standalone, so any figure can be redrawn later,
in any format, on any machine that has the run folder (e.g. pulled from the HF dataset), with no
model or GPU:

```bash
python scripts/make_figures.py runs/M0/smoke_20260912-031000 --format pdf
python scripts/make_figures.py runs/M0/smoke_20260912-031000 --format pdf --out writeup/figs
```

By default figures go to `<run folder>/figures/<format>/` (one folder per format: the run's own PNGs
in `figures/png/`, a PDF redraw in `figures/pdf/`). `--out <folder>` writes straight into that
folder instead. Formats are png/pdf/svg. Which files each milestone's figures read is listed at the
top of `src/jlens_spec/figures.py`. Plotting functions there take DataFrames as read back from
parquet and return a matplotlib Figure, so you can also call them directly in a notebook to restyle
one figure.

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
wire this up already, then `flush()` + `shutdown()` before `io.finalize_run`, which uploads the run
folder one final time (picking up the figures) and then writes and uploads `summary.md` last (see
"When is a run finished?"). Upload failures (e.g. `HF_TOKEN` unset, rate limits) never raise into the GPU loop, but each
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

- **Token boundaries -- decide at M1 with the real tokenizer (Qwen3.6-27B).**
  `tests/test_prompts.py::test_region_tokens_spell_their_text_exactly` is **expected to fail until
  then** (so `pytest tests/` shows 1 failure); any other failure is new. With the stand-in
  tokenizer (Qwen3-0.6B) it finds two things, in all 64 prompts and nothing else:
  - the question's last token is `.\n\n` (period + the blank line before the passage). Decided:
    it stays "question" and is edited (see "Notes for the write-up").
  - sentences 2-5 each start with a token that includes the space before the word (e.g. `" Un"`).
    Decided: accepted as part of that token.

  At M1: run the same check on the real tokenizer. If it also attaches the space to the next word,
  update the sentence spans in `stimuli/stimuli.json` so sentences 2-5 start at that space --
  exactly one space, never two, and no change to the text itself -- and then make the test expect
  exactly those characters.
- **Antonym control departs from invariant 3 on purpose.** To replicate the paper exactly, the
  positive control swaps at every token position, including the first ones that invariant 3 says to
  skip; this applies to that one control only (`configs/experiments/m1_validate.yaml`).
- **Lens file layout** is parsed heuristically (`lens.load_lens`); the download run's
  `lens_resolved.yaml` shows what was found. Also confirm lens keys are block indices (matching `decoder.layers[l]`),
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

## Notes for the write-up

- **The question's last token also carries the paragraph break -- and it is edited (decided
  2026-09-10).** Qwen's tokenizer merges the question's final period with the blank line that
  separates the question from the passage into one token, `.\n\n`. That token is labelled
  "question", so every question-position intervention (all of M3) edits it as well: the edit also
  touches the token that encodes the paragraph break between question and passage. Found in all 64
  prompts by `tests/test_prompts.py::test_region_tokens_spell_their_text_exactly` with the stand-in
  tokenizer (Qwen3-0.6B); to be re-checked with the real tokenizer (Qwen3.6-27B) at M1.

## Secrets

`.env` (not tracked by git) holds `HF_TOKEN` and optionally `GH_TOKEN`. Nothing in this codebase
ever prints, logs, or hardcodes either value.
