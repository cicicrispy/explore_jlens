# explore_jlens -- Stage 0/1 replication

J-lens selectivity replication. The full spec is `coding_prompt_stage01_replication.md`; this README
covers how to run each milestone, and every place the code deliberately departs from the spec
("Departures from the spec", below).

## Status

As of 2026-09-11 (you run everything yourself and report back; the assistant writes the code):

- **M0: done and signed off.** Clean run `runs/M0/smoke_20260911-043447`.
- **M1: in progress.** Token checks with the paper's prompt: `runs/M1/tokens_20260911-183533` (0
  region mismatches). Download on the GPU box (Lambda, 1x H100 80 GB):
  `runs/M1/download_Qwen3.6-27B_20260911-194736` (lens `0731326e…`, 63 layers, width matches).
  First validation `runs/M1/validate_20260911-201047` stopped at the positive control: the swap as
  the spec wrote it undoes itself across the band (each layer flips back what the one before
  flipped). The swap is now **clamped to the clean pass** ("Departures from the spec"). Second
  validation `runs/M1/validate_20260911-203842` (clamped): the answer moves towards 长 with alpha
  but does not flip (大/长 logits: clean 19.0/12.9, alpha 1 18.6/15.6, alpha 2 17.4/17.25). You
  decided to continue: M1 now always runs check 4, check 3 also saves the lens readout on the
  prompt, and M2/M3 need your acceptance in `configs/m1_acceptance.yaml`. Third validation
  `runs/M1/validate_20260911-205658` ran to the end and is **accepted**; bands filled from it
  (workspace [18, 55], full [18, 62], early_late [[18, 33], [45, 55]]). **M2 next.**
- **M2: done.** `runs/M2/loading_20260911-211140` (accuracy 16/16 per question; treatment pair `zh`
  西班牙语 / 法语).
- **M3: in progress.** Controls accepted: `controls_question_20260911-213928`,
  `controls_message_20260911-214953`. `positives_question_20260911-215308` ran: no flips at the
  question positions (the question precedes the passage, so those tokens carry no passage language);
  the `message` set is the main test. Next: `positives_message`, then the anomaly runs.
- **Earlier note (code, stage 3):** Run folders, one file per prompt, uploads as each
  prompt finishes, resume, controls picked by rule in a separate step you review, two position sets,
  a Mac dry run.

See "Known open items" below for things that still need your input.

## Setup

```bash
bash setup.sh
```

Idempotent; detects mac/cuda/cpu-linux, installs torch appropriately, installs the package
editable, (on `cuda` only) downloads the model + lens, runs the test suite (**not on `cuda`**: the
suite runs the stand-in with a random lens, which the Mac covers, and M1's validation checks the
GPU), and writes `requirements-<platform>.lock` (`pip freeze`; git-ignored -- a record of that
machine's packages, never committed).
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
    figures/png/     figures drawn at the end of the run -- only an M3 run's summary figures are
                     uploaded (see "Figures")
    ...              the run's data files (parquet etc.)
    summary.md       the milestone summary -- written LAST, see "When is a run finished?"
```

**When is a run finished? When its `summary.md` is on HF -- nothing else counts.** At the end of
every milestone script (`io.finalize_run`):

1. the run folder is uploaded to the HF dataset (`orbitsoferis/jlens-specificity`), **without
   `figures/`** (except an M3 run's summary figures). `summary.md` doesn't exist yet, so it can't go
   up early. (`hf upload` may split a larger folder into several commits; that's fine.)
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

**Commit before running a milestone.** Records store the git commit, taken **once, when the script
starts** (the code a process runs is the code it loaded, even if files change while it runs). If
code or configs differ from the commit, the stamp is `<commit>-dirty-<fingerprint>`, the
fingerprint being a hash of the uncommitted changes -- so two different sets of edits never share a
stamp.

## Long runs: one file per prompt, uploads as you go, resume (M2, M3)

M2 and M3 save **one set of files per prompt** (`<stimulus>_<question>.parquet` in each of the run's
data folders) and upload each prompt's files **as soon as they are written**, on a background thread
that never blocks the GPU (`io.BackgroundUploader`; upload failures are printed, counted, retried
with the next upload, and reported in the summary; the final folder upload is the backstop).

**Resume.** Running the same command again:
- **resumes** the experiment's **most recent** run if it is unfinished (no `summary.md` on HF) **and
  the code is the same as when it started** (the same commit stamp, uncommitted edits included). A
  prompt whose files are on HF is done and skipped; a prompt whose files exist only on this machine is
  recomputed ("if a file exists on only one machine, it does not exist"); files that are on HF but not
  here (e.g. computed on another machine) are downloaded at the end, before the figures and summary.
- starts a **new** run if the most recent one is finished, or if the code has changed since it
  started (don't `git pull` during a run; after a code change you get a fresh run). An older
  unfinished run is never picked up by default.
- `--fresh` always starts a new run; `--resume runs/M3/<run folder>` resumes exactly that one, even
  after a code change (it says so, and that launch's rows carry the new stamp; the summary lists
  every commit that produced rows). A finished run is never resumed -- it is never changed.
- A resumed run keeps using **its own settings copy**; settings files you changed in `configs/` since
  it started are printed, not applied.
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
  token position across layers 25-75% of depth (each pair clamped in turn to the clean pass's
  coordinates, swapped); 长 should become the top-1 answer instead of 大. Alpha 2
  runs only if alpha 1 fails. If both fail, M1 still runs to the end (check 4 included) and its
  summary says **POSITIVE CONTROL NOT PASSED**; M2 and M3 then refuse that run unless you name it,
  with your reason, in `configs/m1_acceptance.yaml` (`validate_run:` + `reason:`), commit, and
  start them (decided 2026-09-11; the spec: stop). Where the stream actually changed is measured at every
  position, and any change outside the planned positions stops the run; the mask figures
  (`masks/check3_*`) are drawn from those measured changes. The lens readout on the prompt is saved
  too (`check3_readout.parquet`: top-100 at every covered layer and position, clean and under each
  swap; `check3_ranks.parquet`: the ranks of `readout_tokens` from the experiment file), and the
  summary shows it at the last position every 4th layer -- which forms of big/long the lens
  actually reads (added 2026-09-11 after the control's partial result).
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
figures (not uploaded):
- per prompt, `figures/png/loadings/<stimulus>_<question>.png` -- cos(h, v_token) -- and
  `figures/png/ranks/<stimulus>_<question>.png` -- the token's rank in the lens readout, log scale,
  a white line at rank 100: every pair's Spanish and French member, every layer, the workspace
  layers dashed. Each kind uses **one color scale for the whole run**, so a color means the same in
  every prompt's figure;
- `loading_summary_bars`: mean cos per position class and token, averaged over the workspace layers,
  in three rows -- all prompts, the Spanish passages, the French passages.

About 0.5 GB per run, mostly `topk/`. Control tokens are **not** picked in M2 (see M3).

## M3 (Phase B -- after M2)

Per **position set** (which tokens are edited), three steps: a controls run that picks the control
tokens and stops, then a positives run, then an anomaly run:

| Experiment file | Script | Questions | Edits | Run folder |
|---|---|---|---|---|
| `m3_controls_question.yaml` | `m3_controls.py` | (picks controls, no cells) | the question sentence | `runs/M3/controls_question_<time>/` |
| `m3_positives_question.yaml` | `m3_grid.py` | report, hello | the question sentence | `runs/M3/positives_question_<time>/` |
| `m3_anomaly_question.yaml` | `m3_grid.py` | anomaly, content | the question sentence | `runs/M3/anomaly_question_<time>/` |
| `m3_controls_message.yaml` | `m3_controls.py` | (picks controls, no cells) | every token of the user message | `runs/M3/controls_message_<time>/` |
| `m3_positives_message.yaml` | `m3_grid.py` | report, hello | every token of the user message | `runs/M3/positives_message_<time>/` |
| `m3_anomaly_message.yaml` | `m3_grid.py` | anomaly, content | every token of the user message | `runs/M3/anomaly_message_<time>/` |

```bash
python scripts/m3_controls.py --experiment configs/experiments/m3_controls_question.yaml
# read runs/M3/controls_question_<time>/summary.md; if a pick carries language, extend
# controls.control_ineligible in configs/tokens.yaml, commit, and run it again. Then put the accepted
# run's folder name into controls_run: of m3_positives_question.yaml and:
python scripts/m3_grid.py --experiment configs/experiments/m3_positives_question.yaml
# read runs/M3/positives_question_<time>/summary.md -- then, if you decide to continue:
python scripts/m3_grid.py --experiment configs/experiments/m3_anomaly_question.yaml
# if the Hub refuses writes (quota or rate limit), write the run to a folder instead:
python scripts/m3_grid.py --experiment ... --store-dir ~/explore_jlens_store
```

**`--store-dir <folder>` (real runs, `m3_grid.py` only).** The run's store becomes that folder on
this machine: every upload is a file copy there, nothing is committed to the Hub, and `summary.md`
landing there is what makes the run finished. Reads still fall back to the dataset, so the M1, M2 and
positives runs the experiment file names still count as finished. **That folder is then the run's
only backup** -- keep it, and upload the run folder to the dataset when the Hub lets you
(`hf upload orbitsoferis/jlens-specificity runs/M3/<run folder> runs/M3/<run folder> --repo-type
dataset`), after which the run is backed up the normal way and nothing inside it changes. Resume a
`--store-dir` run **with the same `--store-dir`**: a prompt counts as done when its files are in the
store, and that store is the folder.

**Figures split by passage language** (no model, no GPU -- from a finished run's saved records):

```bash
python scripts/make_figures.py runs/M3/positives_message_20260911-221446 --by-language
```

writes `flip_heatmap_es/_fr`, `margin_change_es/_fr` and `margin_vs_deltac_<question>_es/_fr` beside
the run's normal figures, each drawn on the **whole run's** axes so the two languages can be read
against each other. panel c is not repeated -- it already splits by direction × language.

```bash
python scripts/make_figures.py runs/M3/anomaly_message_20260911-230403 --moves
```

writes `margin_moves_<question>` and its `_es` / `_fr` versions: **one panel per intervention kind**,
and inside a panel one line per cell, from that prompt's clean margin (at x = 0, which is what
identity gives) out to the margin the edit ended with, at that cell's mean |Δc|. Colored by passage
language, with the flip boundary dashed at 0. Cells are counted as in every other presentation
figure — one edit each, so the kinds that are the same edit in both directions are drawn once per
passage and label→present twice. Each panel's title carries a p from `scipy.stats.wilcoxon`
(two-sided, `zero_method="zsplit"` so a pair whose edit changed nothing stays in the test) over
exactly the lines drawn in that panel — each line's clean margin paired with its edited margin — with
n = the number of lines.

`--x-size delta_h_norm` puts the residual-stream size on the x axis instead of |Δc| — worth doing
once, because `random_direction` moves outside the swap's 2-D plane and so sits at |Δc| = 0 by
construction.

**The band sweep** (`message` position set, after the four workspace runs): the same experiment over
the other two bands, positives then anomaly in each, so the band is the only difference.

```bash
python scripts/m3_grid.py --experiment configs/experiments/m3_positives_message_full.yaml --store-dir ~/jlens_store
# read its summary, put its folder name into positives_run: of the anomaly file, then:
python scripts/m3_grid.py --experiment configs/experiments/m3_anomaly_message_full.yaml --store-dir ~/jlens_store
# and the same two for ..._early_late.yaml
```

These files carry `reuse_controls_across_bands: true`: they use the **workspace** band's control
tokens (`controls_message_20260911-214953`) instead of picking new ones, so one set of controls runs
in every band. `controls.selection_problems` then ignores `band_layers` -- and nothing else. The cost
is real and is printed in every such run's summary: the picks are no longer size-matched inside the
new band, so their |Δc| ratios drift away from 1.00. Read those ratios before comparing bands.

Fill `from_m2_run` and `m1_validate_run` (and `controls_run` in the positives files, `positives_run`
in the anomaly files) with folder names first. **Hygiene invariant 7's "positive controls first; if
they don't flip, stop"** is your decision between the two runs: the anomaly run refuses to start
unless its positives run is finished and used the same M2 run, band, position set and `skip_first`.
No code applies a threshold.

**Controls are picked by rule, and you read them before any cell runs.** `scripts/m3_controls.py`
picks them and saves them in its run's `controls/` folder (`selection.yaml`, `candidates.parquet`,
`pairs.parquet`), uploads the run and stops. Its summary lists every pick, the next 3 candidates of
each check (what would probably replace a pick you exclude), and every token the exclusion rules
removed. Your lever is `controls.control_ineligible` in `configs/tokens.yaml`: extend it, commit, and
run the controls step again (a new run folder; the rejected one stays as a record). The positives run
names the controls run you accepted (`controls_run:`), refuses it unless it is finished and matches
the positives run's M2 run, band, position set, treatment pair and `skip_first` -- the code version is
**not** compared, so the controls can be picked again after M2 or after other commits -- and copies
its `selection.yaml` into its own `controls/`; the anomaly run reuses that copy. A change of the rules
themselves bumps `SELECTION_VERSION` in `controls.py`, and older selections are refused. The rules are
in `src/jlens_spec/controls.py` (module docstring): candidates come from M2's top-100 readout at the
positions the treatment edits, within the band. There are **16 checks**
(4 treatment rows -- passage language x direction -- x 4 questions; each averages over its 8
passages, the edited positions and the band's layers). A control must reach at least 100% of the
treatment's layer coverage and |Δc|, else at least 75%, else the closest remaining (flagged "below
75%"); among those, the one closest to the treatment wins.
- **label_to_present is picked separately for each check** (a cell uses its own check's 3 tokens),
  and only among tokens that **lower** the removed label in that check (the swap exchanges
  coordinates, so the label becomes the token's coordinate -- lower only if the token's is lower).
  A token that doesn't lower it is used only if a check has too few that do, flagged.
- **big_nonlabel is one set of 3 pairs for every check**, so its rules must hold in all 16. A pair is
  scaled to exactly the treatment's ||Δh||, and after that its |Δc| depends only on how far apart the
  two tokens' lens vectors are compared with the treatment's two -- so pairs are judged on the |Δc|
  they will have after scaling.
Never candidates: the `control_ineligible` tokens and any token containing one of them; the answer
tokens (Yes, No, Hola, Bonjour, with and without a leading space -- one could push an answer
directly); every token without a letter or a digit (punctuation, blank lines, symbols -- odd tokens
that can have large, unspecific effects); and every token that occurs in the passages' own text,
compared without leading spaces and ignoring case (so the passage's ` sous` also rules out `sous` and
` Sous`) -- a word of either language would put a language back in. Tokenizer special tokens and
tokens that don't re-tokenize to
themselves are never controls either; they are listed separately in `selection.yaml`
(`excluded_special`).

Per prompt, both directions: identity, swap (the treatment, alpha 1), random_direction, 3
label_to_present, 3 big_nonlabel = 18 cells; 576 per run. Saved per prompt: `records/` (one row per
cell: margin, flip, top-100 next tokens, full-vocabulary logprobs fp16, intervention logs),
`details/` (per cell x layer x position: where the stream actually changed, next to the planned
log), `tokens/`. **An edit that lands outside its planned positions stops the run.** The summary
reports: the invariant-7 grid check, planned-but-unchanged positions, the two directions' agreement
(identity should be identical; swap, random_direction and big_nonlabel are the same edit in both
directions because the swap is symmetric -- only label_to_present differs), each control's measured
edit size vs the treatment's, per-control flip rates, and 10 sampled raw cells per question x
direction x kind. Figures:
- `panel_c` (the spec's: flip rate with Wilson 95% intervals and mean margin ± SE, per question x
  kind x direction x passage language) and `margin_vs_deltac_<question>`;
- `flip_heatmap`: rows = questions, columns = the five kinds, shade = flip rate, flipped/total in each
  box (swap, big_nonlabel, random_direction, identity count each passage once -- both directions are
  the same edit; label_to_present counts both directions);
- `margin_change`: how far each edit moved the answer -- per question, margin with the edit minus
  the clean margin, in that question's units (e.g. log P(Yes) − log P(No)); bar = mean over
  passages, dots = passages, colored by passage language;
- one mask figure per prompt (5 kinds x 2 directions, from where the stream actually changed).

**The summary figures (everything but the mask figures) are uploaded with the run**; the mask
figures are not. About 0.3 GB per run, mostly the full-vocabulary logprobs.

**Combined figures** (e.g. positives + anomaly of one position set, so all four questions are in one
figure), from the saved records only:

```bash
python scripts/combine_panel_c.py runs/M3/positives_question_20260912-031000 runs/M3/anomaly_question_20260912-041000
```

It writes a new folder `runs/M3/combined_<position set>_<time>/` (`sources.json` + `figures/`:
`panel_c_combined`, `flip_heatmap_combined`, `margin_change_combined`) and, **if every run in it is
finished, uploads it to HF** at the same path (a dry run's goes to its local stand-in,
`runs/dryrun/_hf/`, never to HF). If a run is unfinished, the folder stays local and it says so. It
refuses runs that differ in pair, band, position set, lens or model, and a mix of dry and real runs.
On another machine it needs each run's `records/`, `figure_params.json`, `manifest.json` and
`settings/` (it prints the `hf download` command if they're missing).

## Dry run (Mac, before the GPU)

The whole M2 -> M3 chain on the stand-in model with a **random lens** -- shows the code runs end to
end (per-prompt files, resume, control selection, figures, summaries, combined panel c). The
numbers mean nothing. Nothing is uploaded: `runs/dryrun/_hf/` stands in for the HF dataset; run
folders go to `runs/dryrun/`. The dry runs use their own bands (`configs/experiments/dryrun/
bands.yaml`; `configs/bands.yaml` is untouched) and skip the M1 check. Chain 1 = question set,
band `[9, 18]`; chain 2 = message set, the split band `[[9, 13], [16, 18]]`. Only in dry-run files,
`from_m2_run` / `controls_run` / `positives_run` may say `latest:<experiment name>` (the most recent
finished run), so no hand edits are needed between steps (the dry chain skips the controls review).

```bash
python scripts/m2_loading.py --experiment configs/experiments/dryrun/m2_loading.yaml
python scripts/m3_controls.py --experiment configs/experiments/dryrun/m3_controls_question.yaml
python scripts/m3_grid.py --experiment configs/experiments/dryrun/m3_positives_question.yaml
python scripts/m3_grid.py --experiment configs/experiments/dryrun/m3_anomaly_question.yaml
python scripts/m3_controls.py --experiment configs/experiments/dryrun/m3_controls_message.yaml
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

## Figures (regenerable without the model)

Compute and plotting are separate. Every milestone script saves the data behind each figure to its
run folder (parquet, plus a small `figure_params.json` for the few non-tabular values such as the
band used), and then builds its figures **from those files** via `figures.make_figures` -- never from
in-memory results. **The only figures uploaded to HF are an M3 run's summary figures** (`panel_c`,
`margin_vs_deltac_*`, `flip_heatmap`, `margin_change` -- not its mask figures) **and the combined
figures** (`scripts/combine_panel_c.py`); no other milestone's figures; M0's existing HF files are
left as they are. The data always is, so any figure can be
redrawn later, in any format, on any machine that has the run folder (e.g. pulled from the HF
dataset), with no model or GPU:

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
file's sha256, so it is always the same file. That file holds every weight in one (a "variable"
font), which matplotlib can't use: it would take the lightest weight, draw Chinese Thin and print
`findfont: Failed to find font weight normal, now using 100`. So, the first time, a Regular-weight
copy is made from it with fontTools (which comes with matplotlib) and cached next to it
(`NotoSansSC-Regular.ttf`). Chinese appears in the 27B's M2 figures (the zh pair, '西班牙语' /
'法语') and M1's positive-control figures; the dry run's stand-in drops the zh pair.

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
and clamping its coordinates to targets from one clean pass of the prompt (`interventions.clean_states`;
`runner.run_prompt` records it once per prompt). An M3 experiment file names its band (`band: workspace` for the real M3
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

- **A failed positive control can be accepted in writing** (decided 2026-09-11; the spec: "if both
  fail, stop; do not run M2"): on Qwen3.6-27B the clamped swap moves the answer towards 长 but does
  not flip it, and the paper's results are on Claude models. M1 always runs to the end; M2/M3 accept
  an M1 run whose control did not pass only if `configs/m1_acceptance.yaml` names that run with a
  reason, which their summaries quote (`pipeline.check_m1_passed`).

- **The swap is clamped to the clean pass** (decided 2026-09-11, after M1 run
  `validate_20260911-201047`): the spec's reference loop flips the stream's *current* coordinates at
  every band layer, so each layer undoes the one before (at alpha 1 an even number of layers comes
  back near clean; at alpha 2 the gap grows 3x per layer). Now every layer sets the two coordinates
  to `c_clean + alpha * (flip(c_clean) - c_clean)`, with `c_clean` from one unedited pass of the same
  prompt (`src/jlens_spec/interventions.py`). Two pairs (M1's control) are clamped in turn, each in
  its own basis. The controls are clamps too (big_nonlabel with its own alpha; random_direction sets
  its coordinate along u to the clean value + the size), and every logged size (|Δc|, ||Δh||, what
  controls are matched to) is the clamp's size on the clean pass; what is actually written per layer
  is the `change` column. The paper's released code (github.com/anthropics/jacobian-lens) has no
  swap; its notes call the swap "clamping a lens coordinate ... at every band layer". methods.md
  Section 5.

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
- **Controls are picked by rule in a separate step that stops for your review** (the spec's
  "`controls_proposed.yaml`, approved by the human, copied into `tokens.yaml`" becomes: a controls
  run picks them, you read its summary, extend the blocklist and pick again if needed, and name the
  accepted run in the positives experiment file; `tokens.yaml` keeps only the blocklist; decided
  2026-09-11): matched to the treatment on layer coverage and |Δc| (100%, else 75%, else closest,
  flagged), 3 per kind, candidates from the top **100** readout (the spec's 25 is superseded; its "in
  half the prompts" rule is replaced by the coverage bar), without the answer tokens, punctuation
  (no letter or digit) and the passages' own words (added 2026-09-11).
  label_to_present: 3 tokens per row x question check, each lowering the removed label in its check
  (the spec's "loading below the label" intent); big_nonlabel: one set for all 16 checks, judged on
  the |Δc| it has after scaling to the treatment's ||Δh|| (decided 2026-09-11).
- **Both directions are kept** for every kind, although swap / big_nonlabel / random_direction are
  the same edit in both (the swap is symmetric); the summary reports their agreement.
- **M2 records every lens layer** (the spec: the workspace band) and picks no controls.
- **Figures on HF: only an M3 run's summary figures and the combined figures** (decided 2026-09-11)
  -- no mask figures, no other milestone's figures.
- **Two extra M3 figures**, `flip_heatmap` and `margin_change`, next to the spec's `panel_c` (which is
  unchanged).

- **No tests on the GPU box, and lock files not committed** (decided 2026-09-11): `setup.sh` runs the
  suite on the Mac only (the spec: also on the GPU box, plus GPU-marked tests, of which there are
  none -- M1's validation is the GPU check); `requirements-<platform>.lock` is git-ignored (the spec:
  commit both), since an untracked file would mark every run's code stamp dirty. `nnsight`,
  `transformers` and `accelerate` are pinned in `pyproject.toml` instead.

## Known open items (flagged during writing, need your input)

- **`scripts/m1_tokens.py` re-run with the paper's wrapper:** `runs/M1/tokens_20260911-183533`, 0
  region mismatches on the real tokenizer (instruction regions included). `tests/test_prompts.py::
  test_region_tokens_spell_their_text_exactly` checks the real tokenizer only: the stand-in case (it
  merges the question's final "." with the blank line after it) always failed and was removed
  (decided 2026-09-11) -- a known failure made every test run, and setup.sh, fail.
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
  occur (search for "ASSUMPTION" / "flagged" in `src/jlens_spec/`). The first M1 validation ran the
  27B model and the real lens through loading, the readout and an in-trace edit without errors;
  check 4 (band signatures) has not run on it yet.
- `test_interventions.py`'s check 7 ("apply reproduces the mini-paper's `run` to 1e-4") is skipped
  -- the paper's released code (github.com/anthropics/jacobian-lens) has the lens but no swap, so
  there is no reference to compare against.

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
