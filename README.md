# explore_jlens -- Stage 0/1 replication

J-lens selectivity replication. The full spec is `coding_prompt_stage01_replication.md`; this README
covers how to run each milestone, and every place the code deliberately departs from the spec
("Departures from the spec", below).

## Status

As of 2026-09-11 (you run everything yourself and report back; the assistant writes the code):

- **M0: done and signed off.** Clean run `runs/M0/smoke_20260911-043447`.
- **M1: code written.** The local step has run: `runs/M1/tokens_20260911-055127` -- but that was with
  the OLD prompt (no wrapper); **re-run `scripts/m1_tokens.py`** now that the prompt matches the paper
  (see "Prompt format"). The GPU steps (`download.py`, `m1_validate.py`) have **not** run yet.
- **M2, M3: code written (stage 3), not executed.** Run folders, one file per prompt, uploads as each
  prompt finishes, resume, automatic controls, two position sets, a Mac dry run.

See "Known open items" below for things that still need your input.

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
chat template, single-token checks, chosen controls) are written into the run's folder under
`runs/`; you copy anything that should become canonical into `configs/` by hand.

## Run folders and settings copies

Every run gets its own folder, and nothing is ever overwritten:

```
runs/<milestone>/<experiment name>_<UTC start time>/     e.g. runs/M2/loading_20260912-031000/
    settings/        copies of every settings file the run uses, taken when the run starts
    manifest.json    fixed facts about the run, written once at the start and never changed:
                     run_id, milestone, experiment file, settings sources, config_hash, git
                     commit, start time
    figures/png/     figures drawn at the end of the run -- NEVER uploaded (see "Figures")
    ...              the run's data files (parquet etc.)
    summary.md       the milestone summary -- written LAST, see "When is a run finished?"
```

**When is a run finished? When its `summary.md` is on HF -- nothing else counts.** At the end of
every milestone script (`io.finalize_run`):

1. the run folder is uploaded to the HF dataset (`orbitsoferis/jlens-specificity`), **without
   `figures/`**. `summary.md` doesn't exist yet, so it can't go up early. (`hf upload` may split a
   larger folder into several commits; that's fine.)
2. only after that upload has fully succeeded, `summary.md` is written -- including the run's HF
   folder link and the upload's URL -- and uploaded on its own, as the last step.

If step 1 fails, no summary is written and the error is saved to `upload_error.txt`. If step 2
fails, the summary is renamed `summary_not_uploaded.md` and the error is saved to
`upload_error.txt`. Either way, no `summary.md` means the run is not finished -- on HF and on your
machine. A run stopped mid-upload (e.g. Ctrl-C) also has no `summary.md`. For M2/M3, running the
same command again resumes the run and retries the upload.

**Experiment files.** What a run does is described by a hand-written experiment file in
`configs/experiments/` (e.g. `configs/experiments/m0_smoke.yaml`), passed with `--experiment`.

**Settings copy.** When a run starts, the experiment file and every other settings file the run
reads (e.g. `configs/model.yaml`, `configs/tokens.yaml`, `configs/bands.yaml`, `stimuli/stimuli.json`)
are copied into that run's `settings/` folder. The run then reads its settings **only from that
copy**, never from `configs/`. Editing `configs/` while a run is going, or before resuming it, never
changes that run; the edit applies to the next new run. Code is not copied: every saved row records
the git commit that produced it. `config_hash` in the manifest and in the rows is a hash of the
`settings/` folder. Every saved row also carries the run's `run_id`.

**Commit before running a milestone.** Records store the git commit; if code or configs differ
from it, the hash is suffixed `-dirty`.

## Long runs: one file per prompt, uploads as you go, resume (M2, M3)

M2 and M3 save **one set of files per prompt** (`<stimulus>_<question>.parquet` in each of the run's
data folders) and upload each prompt's files **as soon as they are written**, on a background thread
that never blocks the GPU (`io.BackgroundUploader`; upload failures are printed, counted, retried
with the next upload, and reported in the summary; the final folder upload is the backstop).

**Resume.** Running the same command again:
- **resumes** the experiment's **most recent** run if it is unfinished (no `summary.md` on HF). A
  prompt whose files are on HF is done and skipped; a prompt whose files exist only on this machine is
  recomputed ("if a file exists on only one machine, it does not exist"); files that are on HF but not
  here (e.g. computed on another machine) are downloaded at the end, before the figures and summary.
- starts a **new** run if the most recent one is finished. An older unfinished run is never picked
  up by default.
- `--fresh` always starts a new run; `--resume runs/M3/<run folder>` resumes exactly that one (a
  finished run is never resumed -- it is never changed).
- A resumed run keeps using **its own settings copy**; settings files you changed in `configs/` since
  it started are printed, not applied. Code changes are allowed (every row carries its git commit;
  the summary lists every commit that produced rows).
- A new machine works the same way: the run's settings and manifest are fetched from HF.

**On the GPU box, run inside tmux**, so a dropped SSH connection doesn't kill the run:

```bash
tmux new -s jlens          # start a session (then run the command inside it)
# detach: Ctrl-b then d;   re-attach later:  tmux attach -t jlens
```

If the box dies anyway, start a new one, clone, set up, and run the same command: it resumes.

## M0 (Phase A, Mac -- done)

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

## M1 (lens validation -- only after M0 sign-off)

M1 has three kinds of run, all under `runs/M1/`, each in its own folder and uploaded to HF:

| Step | Where | Command | Run folder |
|---|---|---|---|
| 1. token checks | **your Mac** (tokenizer only, no weights) | `python scripts/m1_tokens.py` | `runs/M1/tokens_<time>/` |
| 2. download | GPU box (`setup.sh` runs it) | `python scripts/download.py` | `runs/M1/download_<model>_<time>/` |
| 3. validation | GPU box | `python scripts/m1_validate.py` | `runs/M1/validate_<time>/` |

**1. Token checks (`configs/experiments/m1_tokens.yaml`).** Loads only the real model's tokenizer and
chat template and records: the region check for all 64 prompts (`region_check.parquet` -- every
region's tokens must spell exactly its text: the question, each sentence, and each piece of
instruction text), the real template string and its diff against the M0 run named in the experiment
file (it now differs by the paper's wrapper -- expected), the single-token table for
`configs/tokens.yaml` (spec M1 step 5), and how the positive-control prompt splits into tokens. Run
it before renting the GPU, so prompts or stimuli can be adjusted first.

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
- readout reproduction on `sp_01` (figure `readout_top1_sp_01`);
- **the Chinese-antonym causal positive control, exactly as in the paper**: the raw prompt
  `"小"的反义词是"` (no chat template), ` big`->` long` and ` bigger`->` longer` swapped at **every**
  token position across layers 25-75% of depth; 长 should become the top-1 answer instead of 大. Alpha 2
  runs only if alpha 1 fails; if both fail, M1 stops (do not run M2 -- M2 and M3 refuse to start
  unless the M1 run you name passed). Where the stream actually changed is measured at every
  position, and any change outside the planned positions stops the run; the mask figures
  (`masks/check3_*`) are drawn from those measured changes.
- band signatures (CKA + next-token agreement + kurtosis by layer, at every non-template position --
  the instruction text included) into `band_signatures.parquet` and `figures/png/band_signatures.png`.
  **You** then fill all three bands in `configs/bands.yaml` (`workspace`, `full`, `early_late`) by
  reading that figure and the CKA heatmap -- no code does this.

## M2 (Phase B -- after `configs/bands.yaml` is filled and M1 passed)

```bash
python scripts/m2_loading.py
```

Settings: `configs/experiments/m2_loading.yaml` -- set `m1_validate_run` to your finished M1
validation run's folder name first. Refuses to start unless that run is finished and its positive
control passed, and `configs/bands.yaml` is complete and consistent (see "Layer bands").

For each of the 64 prompts, **one** clean forward pass (no interventions), at **every** lens layer,
saved to `runs/M2/loading_<time>/`:
- `loadings/` -- cos(h, v_token) and full-readout rank for all 8 language tokens, every position x layer;
- `topk/` -- the lens readout's top-100 tokens at every position x layer (M3's control candidates);
- `answer/` -- the model's answer: full-vocabulary logprobs (fp16), top-100, margin, correct?;
- `tokens/` -- the prompt's tokens and position classes.

Then, from the saved files: `pair_scores.yaml` (every pair's `pair_score` over the workspace band and
the winner -- M3's treatment pair), the accuracy table (in the summary), `checksums.txt`, and the
figures: one grid per prompt (`figures/png/loadings/<stimulus>_<question>.png`: every pair's Spanish
and French member, every layer, the workspace band dashed) plus `loading_summary_bars`. About 0.5 GB
per run, mostly `topk/`. Control tokens are **not** picked in M2 (see M3).

## M3 (Phase B -- after M2)

Four runs -- two **position sets** (which tokens are edited) x {positives, anomaly}:

| Experiment file | Questions | Edits | Run folder |
|---|---|---|---|
| `m3_positives_question.yaml` | report, hello | the question sentence | `runs/M3/positives_question_<time>/` |
| `m3_anomaly_question.yaml` | anomaly, content | the question sentence | `runs/M3/anomaly_question_<time>/` |
| `m3_positives_message.yaml` | report, hello | every token of the user message | `runs/M3/positives_message_<time>/` |
| `m3_anomaly_message.yaml` | anomaly, content | every token of the user message | `runs/M3/anomaly_message_<time>/` |

```bash
python scripts/m3_grid.py --experiment configs/experiments/m3_positives_question.yaml
# read runs/M3/positives_question_<time>/summary.md -- then, if you decide to continue:
python scripts/m3_grid.py --experiment configs/experiments/m3_anomaly_question.yaml
```

Fill `from_m2_run` and `m1_validate_run` (and, in the anomaly files, `positives_run`) with folder
names first. **Hygiene invariant 7's "positive controls first; if they don't flip, stop"** is your
decision between the two runs: the anomaly run refuses to start unless its positives run is finished
and used the same M2 run, band, position set and `skip_first`. No code applies a threshold.

**Controls are picked automatically** (no review) at the start of each positives run and saved in
its `controls/` folder (`selection.yaml`, `candidates.parquet`, `pairs.parquet`); the anomaly run
reuses them. The rules are in `src/jlens_spec/controls.py` (module docstring): candidates come from
M2's top-100 readout at the positions the treatment edits, within the band; each control must reach
at least 100% of the treatment's layer coverage and |Δc| in all four treatment rows (passage
language x direction), else at least 75%, else the closest remaining (flagged "below 75%"); among
those, the one closest to the treatment wins. 3 label_to_present targets + 3 big_nonlabel pairs.
Tokenizer special tokens and tokens that don't re-tokenize to themselves are never controls; they
are listed separately in `selection.yaml` (`excluded_special`). To exclude more tokens, extend
`controls.control_ineligible` in `configs/tokens.yaml`; the next positives run picks it up.

Per prompt, both directions: identity, swap (the treatment, alpha 1), random_direction, 3
label_to_present, 3 big_nonlabel = 18 cells; 576 per run. Saved per prompt: `records/` (one row per
cell: margin, flip, top-100 next tokens, full-vocabulary logprobs fp16, intervention logs),
`details/` (per cell x layer x position: where the stream actually changed, next to the planned
log), `tokens/`. **An edit that lands outside its planned positions stops the run.** The summary
reports: the invariant-7 grid check, planned-but-unchanged positions, the two directions' agreement
(identity should be identical; swap, random_direction and big_nonlabel are the same edit in both
directions because the swap is symmetric -- only label_to_present differs), each control's measured
edit size vs the treatment's, per-control flip rates, and 10 sampled raw cells per question x
direction x kind. Figures: `panel_c`, `margin_vs_deltac_<question>`, and one mask figure per prompt
(5 kinds x 2 directions, from where the stream actually changed). About 0.3 GB per run, mostly the
full-vocabulary logprobs.

**Combined panel c** (e.g. positives + anomaly of one position set), from the saved records only:

```bash
python scripts/combine_panel_c.py runs/M3/positives_question_20260912-031000 runs/M3/anomaly_question_20260912-041000
```

It writes a new **local** folder `runs/M3/combined_<position set>_<time>/` (`sources.json` +
`figures/`), never uploaded -- copy it off the machine yourself (e.g. sftp). It refuses runs that
differ in pair, band, position set, lens or model.

## Dry run (Mac, before the GPU)

The whole M2 -> M3 chain on the stand-in model with a **random lens** -- shows the code runs end to
end (per-prompt files, resume, control selection, figures, summaries, combined panel c). The
numbers mean nothing. Nothing is uploaded: `runs/dryrun/_hf/` stands in for the HF dataset; run
folders go to `runs/dryrun/`. The dry runs use their own bands (`configs/experiments/dryrun/
bands.yaml`; `configs/bands.yaml` is untouched) and skip the M1 check. Chain 1 = question set,
band `[9, 18]`; chain 2 = message set, the split band `[[9, 13], [16, 18]]`. Only in dry-run files,
`from_m2_run` / `positives_run` may say `latest:<experiment name>` (the most recent finished run),
so no hand edits are needed between steps.

```bash
python scripts/m2_loading.py --experiment configs/experiments/dryrun/m2_loading.yaml
python scripts/m3_grid.py --experiment configs/experiments/dryrun/m3_positives_question.yaml
python scripts/m3_grid.py --experiment configs/experiments/dryrun/m3_anomaly_question.yaml
python scripts/m3_grid.py --experiment configs/experiments/dryrun/m3_positives_message.yaml
python scripts/m3_grid.py --experiment configs/experiments/dryrun/m3_anomaly_message.yaml
```

## Prompt format

The user turn is the paper's (transformer-circuits.pub/2026/workspace, Figure 20), from
`configs/prompt_format.yaml`'s `user_message`:

```
I will show you a passage of text. After reading it, answer the following.

{question}

Here is the passage:

{passage}
```

Position classes (`src/jlens_spec/prompts.py`): `question` (the question sentence), `matrix` /
`intrusion` (the passage's sentences), `instruction` (every other character of the user message --
the two instruction sentences and the blank lines between the parts), and `template` (the chat
template around it: `<|im_start|>user`, `<|im_end|>`, the assistant/think prefix -- never edited).
M3's position sets: `question` = {question}; `message` = {instruction, question, matrix, intrusion}.

## Figures (regenerable without the model; never uploaded)

Compute and plotting are separate. Every milestone script saves the data behind each figure to its
run folder (parquet, plus a small `figure_params.json` for the few non-tabular values such as the
band used), and then builds its figures **from those files** via `figures.make_figures` -- never from
in-memory results. **No figure is uploaded to HF** (M0's existing HF files are left as they are): the
data is, so any figure can be redrawn later, in any format, on any machine that has the run folder
(e.g. pulled from the HF dataset), with no model or GPU:

```bash
python scripts/make_figures.py runs/M2/loading_20260912-031000 --format pdf
python scripts/make_figures.py runs/M2/loading_20260912-031000 --format pdf --out writeup/figs
```

By default figures go to `<run folder>/figures/<format>/` (one folder per format). `--out <folder>`
writes straight into that folder instead. Formats are png/pdf/svg. Which files each milestone's
figures read is listed at the top of `src/jlens_spec/figures.py`. Plotting functions there take
DataFrames as read back from parquet and return a matplotlib Figure, so you can also call them
directly in a notebook to restyle one figure.

## Chinese characters in figures

matplotlib's default font (DejaVu Sans) has no Chinese characters and would draw empty boxes (e.g.
in M1's positive-control mask figures, `"小"的反义词是"`). So figures fall back, character by
character, to **Noto Sans SC** (free, SIL Open Font License), which is **downloaded from Google
Fonts' GitHub repo** (`google/fonts`, file `ofl/notosanssc/NotoSansSC[wght].ttf`) the first time a
figure is drawn on a machine, and cached in `$HF_HOME/fonts/`. Nothing is installed and nothing is
stored in git or on HF. The download is pinned to one commit of that repo and checked against the
file's sha256, so it is always the same file.

**If Google moves or renames the file:** update `CJK_FONT_URL` (and `CJK_FONT_SHA256`, the file's
new sha256) at the top of `src/jlens_spec/figures.py`. If the font can't be downloaded (e.g. no
internet), figures use the machine's own Chinese font if it has one (Macs do) and print a warning;
otherwise Chinese characters show as boxes -- the data files are never affected, only the figures.

## Layer bands

**Every `[a, b]` in `configs/bands.yaml` is inclusive on both ends**: `[18, 30]` means layers 18
through 30. You fill all three from M1's figures:
- `workspace` = `[onset, motor_onset - 1]` (M2 and M3 use this one);
- `full` = `[onset, last lens-covered layer]` -- the workspace plus the motor layers, never the final
  layer (Stage 2);
- `early_late` = two blocks, the early and the late part of the workspace (Stage 2).

**M2 and M3 refuse to start unless all three are filled and nest correctly** (`env.check_bands`):
`full` starts at the workspace onset, `workspace` lies inside `full`, both `early_late` blocks lie
inside `workspace`, and every layer is covered by the lens. So the sensory layers below the onset are
never in any band and never edited (M2 still *records* every layer).

`env.expand_band(band_spec)` flattens a pair or a list of pairs into one sorted list of layers.
That flat list is all `interventions.apply` and `runner.py` ever see -- they loop over `layers` in
ascending order inside a single trace and don't care whether those layers are one contiguous run or
several disjoint blocks. So a "swap across layers 18-30 and 40-50" intervention is just
`apply(..., layers=env.expand_band([[18, 30], [40, 50]]), ...)`: one cell, one trace, every layer in
both blocks edited together, each seeing the previously-edited stream from earlier layers in the list
(the "clamped" behavior). An M3 experiment file names its band (`band: workspace` for the real M3
runs; the dry run's chain 2 uses the split `early_late` band to show that works).

## Running a subset of cells (tinkering, not the official M3 runs)

`scripts/m3_grid.py` always builds every kind together for every stimulus/question/direction, per the
spec's hygiene invariant #7 (no treatment cell without its controls in the same run). But
`runner.run_prompt`/`run_cell` don't require every kind in a given call; the only hard requirements
are that `big_nonlabel`/`random_direction` need a `target_norms` from somewhere, and `flip`/
`top1_changed` need a clean baseline from somewhere. Both can come from a previous run's records:

```python
import pandas as pd
from jlens_spec import runner

old = pd.read_parquet("runs/M3/positives_question_20260912-031000/records").to_dict("records")
target_norms_cache = runner.target_norms_cache_from_records(old)   # from the old `swap` rows
cfgs["clean_cache"] = runner.clean_cache_from_records(old)          # from the old `identity` rows
cells = [runner.Cell(..., kind="big_nonlabel", control_tokens=[a_id, b_id], ...) for ...]
records, details = runner.run_prompt(model, lens, cells, cfgs, target_norms_cache)
```

Keys: the clean baseline is keyed by prompt only (`stimulus_id, question_key`), since an identity
pass doesn't depend on direction, pair, layers or alpha. Target norms are keyed by prompt,
direction, pair, layer set and alpha, so a control is only ever scaled against the swap it matches.
A cell with no clean baseline records `flip=None`, `top1_changed=None`, `clean_margin=NaN` --
unknown, not "no effect" -- and `panel_c` leaves those out of the flip rates.

## Departures from the spec (all decided with the human)

- **Prompt = the paper's** (2026-09-11): the wrapper around question and passage, and the paper's
  wording for `report` ("Answer in one word.") and `hello` ("Answer with just that word.") -- logged
  in `stimuli/notes.md`.
- **No "skip the first 4 positions"** from M2 on (`skip_first: 0`, invariant 3): the paper swaps
  across all question tokens, and the high-norm first positions are chat-template tokens, which no
  position set includes. (The antonym control in M1 edits every position, as in the paper.)
- **A second M3 position set, `message`** (every token of the user message), besides the spec's
  question-only set: the paper's text and Figure 20 caption say "across the question tokens", its
  panel labels say "at every position". Both are run.
- **Invariant 7's stop is a human decision between runs**: each position set runs as a positives run
  (report, hello) and, after you read its summary, an anomaly run (anomaly, content).
- **Controls are automatic, with no human approval** (the spec's `controls_proposed.yaml` review step
  is gone; `tokens.yaml` keeps only the blocklist): matched to the treatment on layer coverage and
  |Δc| (100%, else 75%, else closest, flagged), 3 per kind, one set per position set shared by all
  four treatment rows, candidates from the top **100** readout (the spec's 25 is superseded; its
  "in half the prompts" rule is replaced by the coverage bar).
- **Both directions are kept** for every kind, although swap / big_nonlabel / random_direction are
  the same edit in both (the swap is symmetric); the summary reports their agreement.
- **M2 records every lens layer** (the spec: the workspace band) and picks no controls.
- **No figures on HF** for any milestone.

## Known open items (flagged during writing, need your input)

- **Re-run `scripts/m1_tokens.py`** after this stage's code: the prompt now has the paper's wrapper.
  Its region check (real tokenizer) must still show 0 mismatches -- including the new instruction
  regions. The stand-in (`[standin]` case of `tests/test_prompts.py::
  test_region_tokens_spell_their_text_exactly`) may now fail at the wrapper boundaries; that case is
  M0-only and allowed to fail (decided 2026-09-11).
- **Lens file layout** is parsed heuristically (`lens.load_lens`); the download run's
  `lens_resolved.yaml` shows what was found. Also confirm lens keys are block indices (matching
  `decoder.layers[l]`), not hidden_states indices -- an off-by-one there would be silent.
- **`content_probe` -> `content`**: the spec's module contracts mention `content_probe`, but
  `stimuli/stimuli.json`'s `questions` key is `content`. All code here uses `content`.
- **`prompts.build_prompt(stimulus, question_key, fmt)`** needs a tokenizer, the question strings and
  the user-message format, none of which fit the literal 3-argument contract: callers fold them into
  `fmt` via `prompts.make_fmt(tokenizer, stimuli, prompt_format)`.
- Several other module-boundary assumptions (nnsight envoy call semantics outside a trace, the
  `.pt` lens artifact's exact key layout, `runner`'s cell ordering) are documented inline where they
  occur (search for "ASSUMPTION" / "flagged" in `src/jlens_spec/`). None of this has touched a real
  nnsight trace of the 27B model or the real lens file yet, so treat first-run errors there as
  expected, not a sign anything is fundamentally wrong.
- `test_interventions.py`'s check 7 ("apply reproduces the mini-paper's `run` to 1e-4") is skipped
  -- no reference implementation is available in this repo to compare against.

## Notes for the write-up

- **Each of sentences 2-5 starts with a token that includes the space before it** (e.g. `" Un"`);
  the tokenizer attaches the space to the next word, so an edit to a sentence's positions also
  touches that space. The sentence spans in `stimuli/stimuli.json` include it since 2026-09-11.
- M0 (stand-in, old prompt without the wrapper) merged the question's final period with the blank
  line after it into one token `.\n\n`, labelled "question", so M0's smoke cells also touched the
  paragraph break. Only M0 used the stand-in.

## Secrets

`.env` (not tracked by git) holds `HF_TOKEN` and optionally `GH_TOKEN`. Nothing in this codebase
ever prints, logs, or hardcodes either value.
