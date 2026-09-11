# Methods — J-lens selectivity replication (Stage 0/1)

This document describes what the project does and why, in a form meant for a write-up, and keeps a
record of every design decision and every departure from the original specification and from the
paper. It describes the **method**; it reports no scientific results.

**Status (2026-09-11).** All code for milestones M0–M3 is written and tested. M0 has run and is
signed off. M1's tokenizer step has run on a Mac (it must be re-run with the final prompt format);
M1's GPU steps (model and lens download, lens validation), M2 and M3 have **not** run on the real
model yet. The whole M2 → M3 pipeline has been exercised end to end on a Mac "dry run" (a small
stand-in model with a **random** lens), which checks the machinery only: its numbers carry no
information. Section 12 lists what is pending.

Source documents: the project specification (`coding_prompt_stage01_replication.md`), the stimulus
notes (`stimuli/notes.md`), the operational guide (`README.md`) and the code (`src/jlens_spec/`,
`scripts/`). Where this document and the code disagree, the code is authoritative and this document
should be corrected.

---

## 1. Question and approach

The reference is the "workspace" paper (transformer-circuits.pub/2026/workspace), in particular its
Figure 20: a model reads a passage written mostly in one language, with one sentence in another
language, and answers a question about it. Using the **J-lens** — a per-layer linear map that reads
the model's residual stream in vocabulary terms — the paper swaps the lens direction of the
passage's true language with that of an alternative language across the question tokens, and
observes how the model's answers change.

This project replicates that protocol on an open-weights model with an open-source J-lens and asks
how **selective** the swap is:

- **Positive conditions** — questions whose answer depends directly on the passage's language:
  *report* ("What language is this passage written in?") and *hello* ("What is the word for 'hello'
  in the language of this passage?"). If the swap works as a language-label edit, these answers
  should move toward the swapped-in language.
- **Anomaly condition** — *anomaly* ("Does this passage switch languages partway through?"), the
  question of interest.
- **Content probe** — *content* ("Is this passage set outdoors?"), whose answer does not depend on
  language at all; it should not move.
- **Controls** — edits of the same size at the same positions that do not carry the language
  label (Section 6), to separate a label-specific effect from a generic disturbance.

The work is staged. Stage 0/1 (this document) validates the lens, measures how strongly the
language labels are represented ("loading") without any intervention, and runs the paper's protocol
with controls. Later stages (other layer bands, other conditions) reuse the same machinery.

---

## 2. Model and lens

**Model.** `Qwen/Qwen3.6-27B` (Hugging Face, revision `main`), run in bfloat16 on one 80 GB CUDA GPU.
It is a vision-language checkpoint; only its text decoder is used and only text is fed in. Per its
published configuration it has 64 transformer blocks and a hidden size of 5,120 (the lens width is
checked against the model at M1). The residual stream "at layer ℓ" is the output of block ℓ (the
stream after that block), read and written with `nnsight` inside a single forward pass.

**Stand-in model.** `Qwen/Qwen3-0.6B` (28 blocks; same tokenizer family), run in float32 on CPU or
Apple MPS, used for development, the test suite and the dry run. It is never used for results.

**J-lens.** `neuronpedia/jacobian-lens`, file
`qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt`: one d×d matrix J[ℓ] per
source layer ℓ, which transports a residual-stream vector at layer ℓ into the final layer's
coordinates (the final layer's transport is the identity). The exact repository version is resolved
and recorded when it is downloaded (M1).

Definitions used throughout (the spec's reference semantics, implemented as written):

- **Readout** of a hidden vector h at layer ℓ: z = h·J[ℓ]ᵀ, then logits = lm_head(final_norm(z)),
  with the model's own final RMSNorm and unembedding. At the final layer this is the logit lens.
- **Lens vector** of vocabulary token t at layer ℓ: v_t = W_U[t]·J[ℓ], where W_U is the unembedding
  matrix — the direction in layer-ℓ residual space that the lens reads as token t.
- **Loading** of token t at a position and layer: cos(h, v_t), in float32; and the **rank** of t in
  the full readout at that position (1 = the top token).

---

## 3. Stimuli

Sixteen short passages (`stimuli/stimuli.json`), fixed and human-approved before any code ran:

- 8 passages written in **Spanish** containing one **French** sentence, and 8 written in **French**
  containing one **Spanish** sentence. The main language is the *matrix* language; the single
  sentence in the other language is the *intrusion*.
- Each passage has five sentences; the intrusion is always the fourth (second to last). Matrix text
  is 52–62 words, the intrusion 12–16 words.
- Half the passages are set outdoors and half indoors, exactly 4/4 within each matrix language (the
  ground truth for the content probe).
- No proper nouns, and no mention of languages, travel, nationality or food culture. Content words
  in the intrusion sentence avoid Spanish–French cognates; the words replaced, and the moderate
  cognates kept and flagged, are listed in `stimuli/notes.md`.

Four questions are asked about every passage, giving **64 prompts** (16 × 4):

| key | question (verbatim) | correct answer |
|---|---|---|
| report | What language is this passage written in? Answer in one word. | the matrix language |
| hello | What is the word for 'hello' in the language of this passage? Answer with just that word. | *Hola* / *Bonjour* (matrix language) |
| anomaly | Does this passage switch languages partway through? Answer only Yes or No. | Yes (every passage has one intrusion) |
| content | Is this passage set outdoors? Answer only Yes or No. | the passage's `outdoors` field |

*report*, *hello* and *anomaly* use the paper's wording (Figure 20); *content* is not in the paper
and was added by the project's specification.

Stimulus revisions (all logged in `stimuli/notes.md`): one French intrusion word in `sp_01` that is
also a Spanish word (*sur*) was removed; the question key `content_probe` was renamed `content`;
sentence spans 2–5 of every passage were moved one character earlier to include the preceding space,
because both Qwen tokenizers attach that space to the next word (text unchanged); and the *report*
and *hello* wordings were aligned with the paper (2026-09-11).

---

## 4. Prompt construction and token positions

**Prompt.** The user turn is the paper's (Figure 20):

```
I will show you a passage of text. After reading it, answer the following.

{question}

Here is the passage:

{passage}
```

It is wrapped in the model's chat template with thinking disabled, so every prompt ends with the
assistant prefix `<|im_start|>assistant\n<think>\n\n</think>\n\n` (asserted in code). The answer is
read at the last position: the model's next-token distribution there is the answer distribution
("metric position").

**Position classes.** Every token is assigned a class from the tokenizer's character offsets, by
majority character overlap with the regions of the templated text:

- `question` — the question sentence;
- `matrix` / `intrusion` — the passage's sentences in the matrix / intrusion language;
- `instruction` — every other character of the user message (the two instruction sentences and the
  blank lines between the parts);
- `template` — the chat template around the user message (`<|im_start|>user`, `<|im_end|>`, the
  assistant/think prefix). Template tokens are never edited.

Tokens straddling a region boundary are flagged. A region check verifies, for every prompt, that the
tokens assigned to each region spell exactly that region's text; it must pass on the real tokenizer
(it passed with the final prompt in the test suite; the M1 tokenizer run is to be repeated). On the
stand-in tokenizer the question's final period merges with the following blank line into one token,
so the stand-in fails this check; the stand-in is not used for results and this failure is accepted.

**Position sets** (which tokens an M3 intervention edits):

- `question` — the question sentence only (the specification's position set; the paper's text and
  figure caption say the swap is applied "across the question tokens");
- `message` — the whole user message: instruction, question and passage (matrix and intrusion). The
  paper's figure panels are labelled "at every position"; both readings are run.

In this prompt the question comes **before** the passage. The model's states at the question
positions are therefore identical for every passage asked the same question (the model has not yet
read the passage there); an edit at those positions can only reach the answer through later
positions attending back to them. This is one reason the `message` set is also run.

No positions are skipped at the start of the sequence (`skip_first = 0`); the first positions are
chat-template tokens, which no position set includes (a departure from the spec's hygiene
invariant 3, see Section 11).

---

## 5. The swap

All interventions are made inside **one forward pass**, at every layer of a band, in ascending order;
each layer sees the stream already edited at earlier layers ("clamped" swap). Positions outside the
position set are left bitwise unchanged.

**Treatment (swap).** Let s be the token of the language being removed and t the one swapped in, and
V = [v_s, v_t] (d × 2). At each edited position and layer, with c = pinv(V)·h the state's
coordinates along the two lens vectors,

  h ← h + α·V·(flip(c) − c),   α = 1,

where flip exchanges the two coordinates. Everything orthogonal to span(v_s, v_t) is untouched. Two
quantities describe the size of an edit:

- |Δc| = ‖flip(c) − c‖ = √2·|c_s − c_t| — the change in lens coordinates;
- ‖Δh‖ = |c_s − c_t|·‖v_s − v_t‖ — the change in the residual stream.

Both are logged at every edited position and layer, and the change actually produced in the stream
(‖h_new − h‖) is measured at **every** position and layer; any change outside the planned positions
stops the run.

**Treatment pair.** Each language has four surface forms whose first tokens can carry the label
(" Spanish", "Spanish", " spanish", 西班牙语 and the French equivalents), paired by form: *plain*,
*space*, *lower*, *zh*. The pair used in M3 is the one with the highest pair score in the clean pass
(Section 8): the weaker of its two members' mean loading on text in their own language.

**Directions.** *m2i* (matrix → intrusion) removes the passage's matrix language and swaps in the
intrusion's; *i2m* is the reverse. On a French passage, m2i removes " French" and swaps in
" Spanish". Because the swap exchanges both coordinates it is symmetric in (s, t): swap, big
non-label and random-direction cells are the same edit in both directions, and only
label-to-present differs by direction. Both directions are run and their agreement is reported.

---

## 6. Controls

Every treatment cell runs with its controls in the same run (hygiene invariant 7), at the same
positions, layers and prompt:

- **identity** — no edit, through the same code path; the clean baseline every other cell is
  compared with.
- **label-to-present** — the same swap, removing the same label s, but swapping in a *non-language*
  token x that the lens reads at those positions instead of the other language (V = [v_s, v_x]). It
  asks whether weakening the label is enough, or whether putting the other language in is needed.
- **big non-label** — a swap between two strongly present non-language tokens a and b (V = [v_a,
  v_b]); the language coordinates are untouched. Its per-position α is set so that ‖Δh‖ equals the
  treatment's ‖Δh‖ exactly (`norm_scale` = 1), at every position and layer. It asks whether any edit
  this large in the lens's space would have the same effect.
- **random direction** — a random unit vector (fixed per layer and seed), scaled to the treatment's
  ‖Δh‖ at each position. It asks whether any disturbance this large at those positions would.

Three label-to-present tokens (chosen per check, Section 6.1) and three big non-label pairs are used,
each as its own cell.

### 6.1 Automatic control selection

The specification had control tokens proposed by a rule and approved by a human. In this project
they are chosen **automatically, with no human review** (decided with the human), at the start of
each position set's first M3 run, from the clean-pass data:

**Candidates.** Every token among the lens readout's top 100 at the positions the treatment edits,
within the band, on any of the 64 prompts. Excluded: a blocklist of language-identity tokens in any
script (`control_ineligible` in `configs/tokens.yaml`) and any token whose text contains one of them;
tokenizer special tokens and tokens whose text does not re-tokenize to itself (these are listed
separately and never used).

**Measures.** Both are computed within the band, at the edited positions:

- *layer coverage* of a token — the fraction of the band's layers at which it is in the top-100
  readout at one or more edited positions;
- *|Δc|* of a swap — estimated from one clean forward pass.

**Checks.** The treatment has four rows (passage language × direction) and there are four questions,
giving **16 checks** (4 rows × 4 questions). A check averages over its 8 passages, the edited
positions and the band's layers (a ratio of totals — the same average the run later reports for the
measured edit sizes).

**label-to-present — picked separately for each check.** Each check gets its own three tokens, and a
cell uses the tokens of its own check. Because the swap exchanges coordinates, the removed label's
coordinate becomes the control token's: the edit *lowers* the label only if the token's coordinate is
lower (c_x < c_s). So in each check only tokens that lower the removed label on average (mean of
c_s − c_x > 0) are eligible — the intent of the specification's "loading below the label" rule — and
among them the tiers use that check's coverage (≥ the swapped-in token's) and |Δc|(s, x) (≥
|Δc|(s, t)). A token that does not lower the label is used only if a check has fewer than three that
do, and is then flagged.

**big non-label — one set for every check.** The pairs do not touch the label and serve all 16
checks, so their rules must hold in each. Pairs are formed from 20 candidates — those meeting the
coverage bar first, then the widest coverage; each member's coverage must reach the larger of the two
treatment tokens' in every check. Because the edit is
scaled to the treatment's ‖Δh‖, its |Δc| afterwards no longer depends on the state at all: at each
layer it equals (‖v_s − v_t‖ / ‖v_a − v_b‖) × the treatment's |Δc|. Pairs are therefore judged on
the |Δc| they will have **after** scaling, computed from the clean pass and the lens vectors. The
three chosen pairs share no token.

**Tiers and choice.** Tier 1: every ratio (coverage and |Δc|) at least 100% of the treatment's — in
its check (label-to-present) or in every check (big non-label); tier 2: at least 75%; tier 3: the
closest remaining, flagged "below 75%"; for label-to-present, tier 4: does not lower the label,
flagged (selection never stops for lack of a perfect match). Within tiers 1–2 the control closest to
the treatment is preferred: the smallest |log(|Δc| ratio)| (for big non-label, its mean over the
checks). If a treatment token never appears in the top 100 at the edited positions in a check,
coverage sets no bar there, and this is flagged in the run's summary.

The selection, all candidates and all pairs are saved in the run folder (`controls/`); the second
run of the position set reuses them. Each run's summary reports each control's **measured** edit
size relative to the treatment's (|Δc| and ‖Δh‖), since the clean-pass estimate cannot anticipate
the effect of edits at earlier layers of the band.

---

## 7. Measures of the model's answer

All measures use the full next-token distribution at the answer position, never a sampled answer
(hygiene invariant 6). The full log-probability vector of every cell is stored.

- **Margin** (natural log): log P(positive answer) − log P(negative answer), where each side sums
  the probabilities of the first tokens of its answer's forms — e.g. "Yes" and " Yes"; the four
  forms of a language name — (`logsumexp`):
  - *anomaly*, *content*: log P(Yes) − log P(No);
  - *report*: log P(intrusion language's name) − log P(matrix language's name) (names in the four
    forms of Section 5);
  - *hello*: log P(intrusion language's hello) − log P(matrix language's hello).
  Positive values for report and hello mean the answer leans toward the intrusion language, i.e.
  away from the correct answer.
- **Flip**: the side with the higher single best answer-token probability differs from that of the
  clean (identity) cell of the same prompt. A cell without a clean baseline has no flip value and
  is left out of flip counts.
- **Top-1 changed**: the model's overall most likely next token differs from the clean cell's.
- **Margin change**: margin with the edit − clean margin of the same prompt.

Stored per cell: margin, clean margin, flip, top-1 change, the top 100 next tokens, the full
log-probability vector (16-bit), and the per-position, per-layer intervention log.

---

## 8. Pipeline

The work runs as milestones; each ends with its data uploaded, a written summary, and human
sign-off before the next begins.

**M0 — scaffold (Mac, stand-in).** Code, test suite, all 64 prompts built with their position
masks rendered, and one swap and one identity cell with a random lens. Done and signed off.

**M1 — lens validation.**
1. *Tokenizer checks* (Mac, real tokenizer, no weights): the region check for all 64 prompts, the
   exact templated string, which configured answer and language strings are single tokens, and how
   the positive-control prompt tokenizes.
2. *Download* (GPU): model and lens; the lens's repository version, covered layers, dtype, matrix
   shape and coverage are recorded; the run stops if the lens width differs from the model's.
3. *Validation* (GPU), the spec's four checks:
   - final-layer agreement — the lens readout vs the model's actual logits, top-10 overlap at 50
     positions from 5 prompts;
   - readout reproduction — on `sp_01`, at 8 evenly spaced layers, whether a Spanish form is in the
     readout's top 3 at the matrix positions;
   - **causal positive control**, exactly as in the paper — the raw prompt `"小"的反义词是"` ("the
     antonym of 'small' is"), no chat template; the swaps " big"→" long" and " bigger"→" longer" at
     every token position across the lens layers from 25% to 75% of depth; 长 ("long") should
     replace 大 ("big") as the top answer. α = 1, and α = 2 only if α = 1 fails. If both fail, the
     project stops: a null result from an unvalidated lens would be uninterpretable (hygiene
     invariant 1).
   - **band signatures** by layer — linear CKA between layers on the geometry of the lens vectors
     (5,000 tokens sampled uniformly, seed 0); next-token agreement (how often the model's own top-1
     next-token prediction is the lens readout's top-1, or in its top-5, at that layer) at every
     non-template position of the 64 prompts; and the excess kurtosis of the readout logits at the
     same positions. A heuristic
     motor-onset layer is marked: the first layer whose top-1 agreement exceeds the midpoint between
     its median over layers 25–60% of depth and its value at the last layer.

**Layer bands.** Chosen by the human from M1's figures, never by code (hygiene invariant 8); bounds
are inclusive:

- `workspace` = [CKA onset, motor onset − 1] — used by M2 and M3;
- `full` = [CKA onset, last lens-covered layer] — the workspace plus the motor layers (later stage);
- `early_late` = two blocks, the early and the late part of the workspace (later stage).

M2 and M3 refuse to start unless all three are filled and nest (full starts at the workspace onset,
the workspace lies inside full, both early_late blocks lie inside the workspace, every layer is
covered by the lens). The sensory layers below the onset are therefore never edited.

**M2 — clean pass (GPU; no interventions).** One forward pass per prompt, reading **every** lens
layer, saves: the loading (cos) and rank of all language tokens at every position and layer; the
lens readout's top 100 tokens at every position and layer (the control candidates); and the model's
answer (full log-probabilities, top 100, margin, correctness). Then:

- **pair score** of each form pair = the lower of (the Spanish member's mean cos at Spanish text —
  matrix positions of Spanish passages and intrusion positions of French passages — and the French
  member's mean cos at French text), averaged over the workspace layers; the highest-scoring pair is
  M3's treatment pair;
- an accuracy table (answer-set argmax vs ground truth, per question).

**M3 — the paper's protocol with controls (GPU).** Four runs, two position sets × two runs:

| run | questions | edits |
|---|---|---|
| positives, question | report, hello | the question sentence |
| anomaly, question | anomaly, content | the question sentence |
| positives, message | report, hello | the whole user message |
| anomaly, message | anomaly, content | the whole user message |

Band: workspace; α = 1; both directions. Per prompt and direction: identity, swap, random direction,
3 label-to-present and 3 big non-label cells (9 cells; 18 per prompt), so 576 cells per run (16
passages × 2 questions × 2 directions × 9). The positives run of a position set runs first and picks
the controls; the human reads its summary and only then launches the anomaly run (hygiene
invariant 7's "if the positive controls don't flip, stop" — a human decision, no threshold in code).
The anomaly run refuses to start unless its positives run is finished and used the same M2 run,
band, position set and settings.

Each M3 run's summary reports, by assertion: the number of cells; that no prompt × direction group
is missing a cell; that no edit landed outside the planned positions; planned positions whose stream
did not change; the agreement of the two directions (identity cells must be identical; the symmetric
edits must agree up to 16-bit storage rounding); each control's measured edit size relative to the
treatment's; per-control flip rates and mean margins; and ten randomly sampled raw cells per question
× direction × kind, with the clean and intervened top-5 tokens.

---

## 9. Figures and statistics

Every figure is drawn from the saved data files, never from in-memory results, and can be redrawn
without the model.

- **M1:** the causal positive control's masks; the readout top-1 grid; the CKA heatmap; the band
  signatures.
- **M2:** per prompt, the loading (cos) and the rank of each pair member at every position × layer,
  with the workspace layers marked (rank on a log scale, with a line at rank 100), each on one color
  scale for the whole run; and the mean loading per position class and token over the workspace
  layers, for all prompts and for the Spanish and the French passages separately.
- **M3:**
  - *panel c* (the spec's version of the paper's panel c): flip rate with Wilson 95% intervals, and
    mean margin ± standard error, per question × kind × direction × passage language;
  - *margin vs |Δc|*: every cell's margin against its mean |Δc|;
  - *flip heatmap*: questions × kinds, shade = flip rate, each box giving flipped/total;
  - *margin change*: per question, the mean margin change of each edit over passages, with every
    passage as a point, colored by passage language;
  - per prompt, where the stream actually changed, for every kind and direction;
  - combined versions of panel c, the flip heatmap and the margin change over a position set's two
    runs (all four questions).

**Counting.** Each bar or box pools the passages (and, for the two pooled controls, their three
cells). In the flip heatmap and margin-change figures, the kinds that are the same edit in both
directions count each passage once; label-to-present counts both directions. Flip rates are reported
as counts (k of n); resting on 8–16 passages, they are coarse, and panel c's Wilson intervals show
that. No hypothesis tests are run at this stage.

---

## 10. Reproducibility and data management

- **Run folders.** Every run gets its own folder, `runs/<milestone>/<experiment>_<UTC start time>/`,
  never reused or overwritten, with a copy of every settings file it reads; the run reads only that
  copy. A manifest records the run's identity, settings sources, a hash of its settings, the code
  version and the start time.
- **Code version.** Every stored row carries the git commit of the code that produced it, taken once
  when the script starts. Uncommitted changes are recorded as `<commit>-dirty-<fingerprint>`, the
  fingerprint being a hash of the changes, so two different sets of edits never share a stamp.
- **Storage.** The only store is a private Hugging Face dataset (`orbitsoferis/jlens-specificity`).
  Long runs (M2, M3) save one set of files per prompt and upload each as soon as it is written. A run
  counts as finished only when its summary is in the dataset; the summary is written and uploaded
  last, after everything else has uploaded. Checksums (SHA-256) of every data file are recorded.
- **Resuming.** Re-running the same command resumes the most recent run of that experiment if it is
  unfinished **and** the code is unchanged since it started; after a code change a new run starts. A
  resumed run keeps its original settings copy.
- **Precision.** The real model runs in bfloat16. Margins and flips are computed in 32-bit before
  storage; full log-probability vectors are stored in 16-bit (a storage step of up to ~0.03 at the
  magnitudes involved).
- **Figures in the dataset.** The data behind every figure is uploaded. Of the figures themselves,
  only M3's summary figures and the combined figures are uploaded; the rest are drawn locally.
- **Software.** Python ≥ 3.11, PyTorch, Hugging Face `transformers`, `nnsight` ≥ 0.8.0rc1 for reading
  and writing activations; pandas/pyarrow (Parquet) for data; matplotlib for figures (with a pinned,
  checksummed Noto Sans SC font for Chinese characters).
- **Tests.** A test suite (CPU, stand-in model, random lens) covers the swap mathematics (e.g. zero
  effect when v_s = v_t, the orthogonal complement unchanged, linear scaling in α, swap ≠ steering),
  the scaling of the controls, prompt construction and position classes, the measures, control
  selection, storage and resume logic, and the figures.
- **Dry run.** Before any GPU time, the full M2 → M3 chain (one M2 run, four M3 runs, the combined
  figures) was run on a Mac with the stand-in model and a random lens, with a local folder standing in
  for the dataset. It verified the machinery (576 cells per run, no edits outside the planned
  positions, identity cells identical across directions, the symmetric edits' agreement within
  16-bit storage precision, control selection, figures); its numbers are meaningless. Resuming is
  covered by the test suite.

---

## 11. Departures from the specification (all decided with the human)

| topic | specification | this project | why |
|---|---|---|---|
| Prompt | user turn = question + passage | the paper's wrapper (Section 4); the paper's wording for *report* and *hello* | match the paper exactly (2026-09-11) |
| Skipped start positions | skip the first ~4 positions (invariant 3) | none (`skip_first` = 0) | the paper swaps across all question tokens; the first positions are template tokens, never edited |
| M1 positive control | — | edits every position (incl. the first), raw prompt | exactly as in the paper; this control only |
| M3 position sets | question only | question **and** message | the paper's text says "across the question tokens", its figure panels "at every position"; both run |
| Invariant 7's stop | positives before anomaly; stop if they don't flip | positives and anomaly as separate runs; the human reads the positives summary before the anomaly run | a human decision, no threshold |
| Control tokens | proposed by rule, approved by a human | chosen automatically, no review (Section 6.1) | decided with the human |
| Control pool | top-25 readout at question positions, frequency ≥ 0.5 | top-100 at the edited positions; a coverage bar replaces the frequency rule | wider pool; matched on coverage and size |
| Control matching | label-to-present: the most frequent eligible token loading below the label; big non-label: the two eligible tokens with the highest mean loading | matched to the treatment on coverage and \|Δc\|: label-to-present per row × question check, among tokens that lower the label there; big non-label one set for all 16 checks, judged on its \|Δc\| after scaling | controls should do what the treatment does, in size and reach |
| Directions | m2i and i2m | both kept, though swap / big non-label / random are the same edit in both | the literal grid; agreement reported |
| M2 layers | the workspace band | every lens layer (bands still used for scores) | record everything once |
| Stored top tokens | top-5 | top-100 everywhere | any smaller k can be applied later |
| Figures on the dataset | (not specified; M0's upload included its figures) | only M3's summary figures and the combined figures | the data is uploaded; figures are redrawable |
| Extra figures | panel c, margin vs \|Δc\| | also flip heatmap, margin change, M2 rank heatmaps | presentation |
| Question key | `content_probe` | `content` | naming |

Two further operational choices: secrets are loaded only from Python (never by sourcing the secrets
file in a shell), and a run is resumed only if the code is unchanged since it started.

---

## 12. Pending and open items

- Re-run the M1 tokenizer checks with the final prompt; the region check must show no mismatches.
- M1 on the GPU: download the model and lens, record the lens version, run the validation. **If the
  causal positive control fails at α = 1 and α = 2, the project stops.**
- The human fills the three layer bands from M1's figures.
- M2 and the four M3 runs on the real model.
- Confirm on the GPU that the lens file's layer keys are block indices (matching the decoder's
  layers), not hidden-state indices — an off-by-one there would be silent.

---

## Appendix A. Decision log

All dates 2026-09-11 unless noted.

- M0 signed off (`runs/M0/smoke_20260911-043447`).
- Stimulus spans moved to include the preceding space (after M1 tokenizer run
  `tokens_20260911-054336`).
- Prompt and question wording aligned with the paper; new `instruction` position class.
- Stage-3 design agreed: two position sets; positives/anomaly split runs; automatic controls from
  the top-100 readout, matched on coverage and |Δc| with 100% / 75% / closest tiers; M2 records every
  lens layer and saves the top-100 readout (the residual stream itself is not saved); no human review
  of controls; both directions kept; figures redrawn from data.
- A YAML quirk found by the dry run: unquoted `yes` / `no` keys load as booleans; the answer keys are
  now quoted, with a clear error if not.
- Full log-probabilities kept in 16-bit storage; the two swap directions are checked to agree within
  one 16-bit step (and, in 32-bit, to 10⁻³ in the tests).
- After the first dry M3 run: big non-label pairs judged on their |Δc| after scaling, averaged across
  layers like label-to-present; every control rule checked per row × question (16 checks).
- While writing this document: the size-only rule could pick a label-to-present token that *raises*
  the label (|Δc| does not see the direction). Label-to-present is now picked separately for each of
  the 16 checks, among tokens that lower the removed label in that check.
- Figures: M3 summary and combined figures uploaded to the dataset, mask figures and M2 figures not;
  added the flip heatmap, the margin change and the M2 rank heatmaps; panel c kept exactly as
  specified.
- Code stamp taken once per launch with a fingerprint of uncommitted changes; a run is resumed only
  if the code is unchanged.
