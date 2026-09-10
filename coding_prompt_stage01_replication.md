# Coding Agent Prompt — Stage 0/1: Lens validation, workspace loading, paper-protocol replication

You are the engineer on a mechanistic interpretability replication. You write code, run it, produce artifacts, and report. You do not interpret results scientifically and you do not change the experimental design; when something is ambiguous or fails, you stop and report.

## Repos and inputs

- **Code (GitHub):** `https://github.com/cicicrispy/explore_jlens`. Work only on branch `stage01/replication`; create it from `main` at the start. Commit and push to that branch at every milestone. **Never commit to, merge into, rebase onto, or open a pull request against `main`** — the human reviews the branch and merges it themselves after sign-off. If `git` reports you are on `main`, stop and switch branches before doing anything else.
- **Artifacts (Hugging Face dataset):** `orbitsoferis/jlens-specificity` — private. Large outputs go here, never into git.
- **Stimuli:** already in the repo at `stimuli/stimuli.json` and `stimuli/notes.md`. Human-approved. **Read-only.** Do not edit, reformat, or "fix" them. If something in them blocks you, stop and report.

## Secrets — read this before writing any code

A `.env` file will be placed in the repo root by the human on each machine. It is not yours to inspect.

- **Never** `cat`, `less`, `head`, `echo`, `print`, log, `grep`, or otherwise read or display `.env` or any environment variable value. To check a variable is set: `[ -n "$HF_TOKEN" ] && echo set || echo unset`, nothing else.
- Load secrets only via `set -a; source .env; set +a` in `setup.sh`. `hf`, `huggingface_hub`, `transformers`, and `gh` read `HF_TOKEN` / `GH_TOKEN` from the environment automatically — never pass tokens as arguments or CLI flags, never put them in URLs or `.git/config`.
- `.env` is in `.gitignore` in the **first** commit. `git check-ignore .env` before every push.
- Do not run `env`, `printenv`, or anything else that dumps environment values.

## Two environments

**Phase A — local, Apple Silicon Mac, no CUDA.** M0 happens here entirely: all code, unit tests, prompt construction, position classification, figure code, logging, against a small stand-in model on CPU. (MPS may be used opportunistically but every test must pass with `device=cpu`.) `setup.sh` detects the platform and skips model/lens download and GPU steps.

**Phase B — remote Linux box, one 80 GB CUDA card.** M1 onward. The human provisions it, clones the repo, places `.env`. You run `setup.sh`, which now installs CUDA torch, downloads the 27B model and the lens, and re-runs the full test suite before any milestone job.

**State moves only through git and the HF dataset repo.** If a file exists on only one machine, it does not exist. Every `summary.md` states which environment it ran in.

Do not attempt to download the 27B model or the 3.3 GB lens in Phase A.

## Packaging

`pyproject.toml` with:
- core deps: `nnsight>=0.8`, `transformers`, `huggingface_hub`, `pyyaml`, `pandas`, `pyarrow`, `matplotlib`, `numpy`, `pytest`.
- `torch` is **not** pinned in `pyproject.toml`; `setup.sh` installs it per platform: on macOS `pip install torch` (CPU/MPS wheel); on Linux with `nvidia-smi` present, the CUDA wheel from the official index matching the driver's CUDA version (detect via `nvidia-smi`; if unsure, use the default `pip install torch` CUDA build and record what was installed).
- Pin everything you install by writing `pip freeze > requirements-<platform>.lock` at the end of setup and committing both lock files.

`setup.sh` (idempotent; exits non-zero on any failure):
1. Create/activate `.venv` (Python ≥3.11).
2. Detect platform: `PLATFORM=mac` if `uname -s` = Darwin; `PLATFORM=cuda` if `nvidia-smi` succeeds; else `PLATFORM=cpu-linux`.
3. Install torch per platform, then `pip install -e .`.
4. Install `hf` CLI (`pip install -U huggingface_hub[cli]`) and run `hf skills add --claude`. Install `gh` if absent (brew on mac, apt on linux).
5. `set -a; source .env; set +a`; `gh auth setup-git`.
6. Export `HF_HOME` to `$REPO/.hf_cache` on mac or the large disk on the GPU box (read from `configs/paths.yaml`).
7. On `cuda` only: download model and lens (`scripts/download.py`).
8. `pytest tests/ -q`. On `cuda`, additionally `pytest tests/ -q -m gpu`.
9. Write `requirements-$PLATFORM.lock`; print `SETUP OK ($PLATFORM)`.

## Repo layout (build exactly this)

```
explore_jlens/
  pyproject.toml
  setup.sh
  .gitignore                 # .env, .venv, .hf_cache, runs/**/*.parquet, runs/**/*.pt
  .env.example               # variable NAMES only, no values
  README.md                  # how to run each milestone
  stimuli/                   # provided; read-only
    stimuli.json
    notes.md
  configs/
    paths.yaml               # hf_home per platform, runs dir
    model.yaml               # hf_id, revision, dtype, standin_hf_id
    lens.yaml                # repo, filename, resolved revision sha, metadata dumped from the .pt
    prompt_format.yaml       # exact template string, think-prefill, metric-position rule
    tokens.yaml              # language token candidates, control targets, hello answers
    bands.yaml               # layer bands; filled by HUMAN after M1 (leave a placeholder)
  src/jlens_spec/
    __init__.py
    env.py
    model.py
    lens.py
    prompts.py
    interventions.py
    loading.py
    metrics.py
    runner.py
    figures.py
    cka.py
    io.py
  scripts/
    download.py
    m0_smoke.py
    m1_validate.py
    m2_loading.py
    m3_grid.py
  tests/
    conftest.py              # tiny stand-in model fixture + random lens fixture
    test_env.py
    test_lens.py
    test_prompts.py
    test_interventions.py
    test_loading.py
    test_metrics.py
    test_cka.py
    test_io.py
  runs/
    M0/ M1/ M2/ M3/          # each: summary.md, figures/, *.parquet, manifest.json
```

## Applying the J-lens (reference semantics — implement these, do not re-derive)

The lens artifact gives one matrix `J[ℓ]` (d×d) per source layer ℓ. It transports a residual-stream vector at layer ℓ into the final layer's coordinates. The final layer itself is not in the artifact; its transport is the identity.

**Where `h` comes from.** `h_ℓ = model.model.layers[ℓ].output` inside an `nnsight` trace: the block output, i.e. the residual stream *after* layer ℓ, shape `[batch, pos, d]`. This is what the artifact was fit on (raw HF block outputs). Do not use pre-norm inputs, attention or MLP sub-outputs, or any TransformerLens hook name.

**Reading (lens readout).** For a hidden vector `h` at layer ℓ:
```
z      = h.float() @ J[ℓ].T                # transport to final-layer coordinates
logits = lm_head( final_norm(z) )          # the model's OWN final RMSNorm and unembedding
```
For Qwen: `final_norm = model.model.norm`, `lm_head = model.lm_head`. Qwen has **no** logit soft-cap; do not copy the Gemma `tanh` from the mini-paper. Cast `z` to the `lm_head` weight dtype before the norm. Top-k over `logits` is "the readout." At the final layer skip the transport (`J = I`) — this is the logit lens, and the two must agree there (M1 check 1).

**Lens vectors.** The direction in layer-ℓ residual space attached to vocabulary token `t`:
```
v_t^(ℓ) = W_U[t] @ J[ℓ]                    # W_U = model.lm_head.weight, shape [vocab, d]
```
so `lens_vectors(tokens, ℓ) = W_U[token_ids].float() @ J[ℓ]` → `[n_tokens, d]`. Note `J[ℓ]` not `J[ℓ].T` here — this is the row of `W_U J` pulled back to layer ℓ (mini-paper convention).

**Per-token probe / loading.** `cos(h, v_t)` with both in float32. Rank of `t` = `1 + (logits > logits[t]).sum()` over the full readout.

**Writing (swap in lens coordinates), inside a single trace, clamped across a layer band:**
```python
with model.trace(prompt.input_ids):
    for l in layers:                                   # ascending; each layer sees the edited stream
        env = model.model.layers[l].output             # [batch, pos, d]
        h   = env.float()
        V, V_pinv = basis[l]                           # V: [d, 2] = [v_s, v_t]; V_pinv: [2, d]
        c   = h @ V_pinv.T                             # [batch, pos, 2]
        delta = alpha * ((c[..., [1, 0]] - c) @ V.T)   # [batch, pos, d]
        delta = delta * mask[None, :, None]            # zero outside intervened positions
        model.model.layers[l].output[:] = (h + delta).to(env.dtype)
    logits = model.output.logits[0, prompt.metric_pos].float().save()
```
`basis[l] = (V, torch.linalg.pinv(V))` is precomputed per layer from `lens_vectors([s, t], l)`. Log `c` before, `c` after, `‖delta‖` per masked position from the same trace (save them). `label_to_present` uses the same block with `V = [v_source, v_target]`; `big_nonlabel` with `V = [v_a, v_b]` and per-position `alpha`; `random_direction` replaces `delta` with `target_norm[pos] * u_l` for a fixed random unit `u_l`; `identity` sets `delta = 0` but still runs the trace so logits come from the same code path.

**What you never do:** steer (`h + α v_t`) in any treatment or control cell; write to the final layer; swap at template/special positions; apply the swap outside a trace (nnsight's in-place assignment on the envoy is what makes the edit propagate).

## Module specifications

Types below are the contract. Docstrings must state units and shapes.

### `env.py`
- `get_device() -> torch.device` — `cuda` if available, else `cpu` (never default to `mps` for tests).
- `require_env(name: str) -> None` — raises if unset; never returns or logs the value.
- `git_commit() -> str`, `config_hash(*paths) -> str` (sha256 of concatenated YAML contents).

### `model.py`
- `load_model(cfg: dict, standin: bool = False) -> nnsight.TransformersModel` — loads `cfg["hf_id"]` at `cfg["revision"]` in bf16 with `device_map="auto"`, or `cfg["standin_hf_id"]` (a small Qwen3-family model, ≤1B params, same tokenizer family) when `standin=True`. Asserts the tokenizer is Qwen-family. Returns the nnsight wrapper.
- `n_layers(model) -> int`, `d_model(model) -> int`, `layer_output(model, l)` — the envoy for `model.model.layers[l].output` (block output = residual stream after layer `l`).
- `unembed(model, h: Tensor[..., d]) -> Tensor[..., vocab]` — the model's own final norm + `lm_head`, matching the mini-paper.

### `lens.py`
- `load_lens(cfg: dict, device) -> Lens` where `Lens` is a dataclass: `J: dict[int, Tensor[d,d]]`, `d_model: int`, `meta: dict` (every non-tensor key in the `.pt`, verbatim), `layers: list[int]`, `dtype: torch.dtype`, `file_size_bytes: int`. Loaded with `torch.load(..., map_location="cpu", weights_only=True)`. For tests, `random_lens(d, layers, seed) -> Lens`.
- `lens_vectors(model, lens, tokens: list[str|int], layer: int) -> Tensor[len(tokens), d]` — `W_U[token_ids] @ J[layer]` (mini-paper convention). Raises if any string is not a single token.
- `readout(model, lens, h: Tensor[pos, d], layer: int, k: int = 10) -> (Tensor[pos,k] ids, Tensor[pos,k] logits)` — transport-then-unembed, top-k.
- `token_rank(model, lens, h, layer, token_id) -> Tensor[pos]` — 1-based rank of `token_id` in the full readout.
- `coverage_ratio(lens) -> float` — `file_size_bytes / (len(layers) * d_model**2 * bytes_per_elem)`.

### `prompts.py`
- `PositionClass = Literal["template", "question", "matrix", "intrusion"]`
- `build_prompt(stimulus: dict, question_key: str, fmt: dict) -> Prompt` where `Prompt` is a dataclass: `text: str` (fully templated string incl. think-prefill), `input_ids: list[int]`, `classes: list[PositionClass]` (one per token), `metric_pos: int`, `stimulus_id: str`, `question_key: str`, `spans: dict` (char spans of question and each sentence *within the templated text*), `flags: list[str]` (e.g. tokens straddling a boundary).
  - Template: user turn = `f"{question}\n\n{passage}"`. Build with `tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=False)`. The result **must end with the literal suffix** `<|im_start|>assistant\n<think>\n\n</think>\n\n` (this is what the reference observation used; the answer token follows it directly). Assert this. If the tokenizer does not accept `enable_thinking`, build with `add_generation_prompt=True` and append `<think>\n\n</think>\n\n` manually, and say so in `summary.md`. Record the exact final string for one stimulus in `configs/prompt_format.yaml` at M0 (stand-in tokenizer) and **re-derive and re-record it at M1 from the real tokenizer**; the M1 string is canonical, and `summary.md` shows the diff if any. Tests assert the suffix and the metric-position structure, never a hardcoded full string.
  - Classes come from `return_offsets_mapping=True`; a token is `question`/`matrix`/`intrusion` by majority character overlap with the corresponding span, else `template`. Straddling tokens are flagged.
  - `metric_pos` = index of the first token after the think prefill (i.e. where the assistant's answer token is predicted: `len(input_ids) - 1`). Assert `classes[metric_pos] == "template"` and that the preceding tokens are the prefill.
- `mask(prompt: Prompt, classes: set[PositionClass], skip_first: int = 4) -> Tensor[pos] bool`.
- `render_mask(prompt: Prompt, mask: Tensor, path: Path) -> None` — PNG: tokens as boxes, colored by class, intervened tokens outlined; the metric position marked.

### `interventions.py`
All take `h: Tensor[batch, pos, d]` and a `mask: Tensor[pos] bool` and return a new tensor; unmasked positions are bitwise unchanged. Each also returns an `InterventionLog` dataclass per masked position: `pos, layer, c_before: (float,float), c_after: (float,float), delta_c_norm: float, delta_h_norm: float, alpha: float, kind: str`.
- `swap(h, mask, v_s: Tensor[d], v_t: Tensor[d], alpha: float = 1.0) -> (Tensor, list[InterventionLog])` — `V=[v_s,v_t]` (d×2), `c = pinv(V) @ h`, `h + alpha * V @ (flip(c) - c)`.
- `label_to_present(h, mask, v_source, v_target, alpha=1.0)` — same math as `swap`, **same source token as the treatment** (the label being removed), target = `controls.label_to_present.target`, a non-label token present at those positions with loading below the label's. Removes the label and installs a present non-label concept. Because the swap exchanges coordinates, a target with loading ≥ the label's would *not* remove the label — hence the modest-loading requirement. Separates "the circuit needs the label present" from "the circuit reads the label's content."
- `big_nonlabel(h, mask, v_a, v_b, target_norms)` — swap between two strongly-present **non-label** tokens (`controls.big_nonlabel.pair`), label coordinate untouched, with per-position `alpha` set so `delta_h_norm == norm_scale * target_norms[pos]` (`norm_scale ≥ 1`). An in-J-space edit at least as large as the treatment, at the same positions, that does not touch the label. If this leaves anomaly unmoved while `swap` flips it, the flip is not a magnitude artifact.
- `random_direction(h, mask, target_norms, seed) -> ...` — add a random unit vector (fixed per seed, per layer) scaled to `target_norms[pos]` = the treatment cell's per-position `delta_h_norm`. This is the only control that does not touch the label coordinate; it tests generic disturbance at those positions.
- `identity(h, mask) -> ...` — no-op, logs zeros.
- `apply(model, lens, prompt, kind: str, layers: list[int], mask, **kw)` — registers the intervention on `layer_output(model, l)` for each `l` in `layers` inside one `model.trace(...)`, clamped (each layer sees the already-edited stream), and returns `(logits_at_metric: Tensor[vocab], logs: list[InterventionLog])`.

### `loading.py`
- `loading(model, lens, prompt, token_ids: list[int], layers: list[int]) -> DataFrame` with columns `stimulus_id, question_key, pos, class, layer, token, token_id, cos, rank` — `cos = cos(h[pos], v_token(layer))`, `rank` from `token_rank`. One clean forward pass per prompt; all layers read in the same trace.
- `pair_score(df, pair: tuple[str,str], band: list[int]) -> float` — `min(mean cos of Spanish member over Spanish-language positions, mean cos of French member over French-language positions)`, averaged over `band`. "Spanish-language positions" = matrix positions of es-matrix passages plus intrusion positions of fr-matrix passages, and symmetrically.

### `metrics.py`
- `margin(logprobs: Tensor[vocab], pos_ids: list[int], neg_ids: list[int]) -> float` — `logsumexp(pos) - logsumexp(neg)`.
- `answer_ids(tokenizer, answers: list[str]) -> list[int]` — first token of each answer, with and without leading space, deduplicated. Used for `Yes/No`, `Spanish/French` variants, `Hola/Bonjour`.
- `question_margin(question_key, logprobs, tokens_cfg, matrix_lang) -> float` — dispatches: `anomaly`/`content_probe` → Yes vs No; `report` → target-language names vs source-language names; `hello` → target hello vs source hello. Sign convention documented: positive = "answer as if the label were the *target* language" for report/hello; positive = Yes for anomaly/content.

### `runner.py`
- `Cell` dataclass: `stimulus_id, question_key, direction ("m2i"|"i2m"), pair_name, kind, layers, position_set, alpha, seed`.
- `run_cell(model, lens, cell, cfgs) -> CellRecord` where `CellRecord` has: everything in `Cell`, `margin: float`, `clean_margin: float` (from the matching `identity` cell), `flip: bool` (argmax over the question's answer set differs from the clean cell), `top1_changed: bool` (the model's overall top-1 token differs from clean), `logprobs_fp16: np.ndarray[vocab]`, `top5: list[(str,float)]`, `intervention_logs: list[InterventionLog]`, `prompt_len: int`, `metric_pos: int`, `git_commit, config_hash, lens_sha, model_revision, timestamp`.
- `run_grid(cells: list[Cell], ...) -> Iterator[CellRecord]` — batches prompts of equal length where possible; writes records incrementally to parquet (`io.append_records`) so a crash loses at most one batch.

### `figures.py`
- `mask_figure` (delegates to `prompts.render_mask`), `loading_heatmap(df, stimulus_id, token) -> PNG`, `loading_summary_bars(df, band) -> PNG` (every variant side by side, grouped by position class × matrix language), `cka_matrix(...)` (below), `panel_c(records) -> PNG` (two rows: top = flip rate with Wilson 95% CI per question, treatment vs 4 controls, split by direction × matrix language — this is the paper's panel c; bottom = mean margin ± SE for the same cells), `margin_vs_deltac(records, question_key) -> PNG`.

### `cka.py` (add to `src/jlens_spec/`)
Linear centered kernel alignment between layers, on the geometry of the lens vectors. This is how the workspace band is chosen (paper §struct-layers): blocks of high mutual similarity = layers sharing a representational geometry.
- `cka_matrix(model, lens, n_tokens: int = 5000, seed: int = 0) -> (Tensor[L, L], list[int] layers)`:
  1. Sample `n_tokens` token ids uniformly from the vocabulary with `seed`; use the **same** sample for every layer.
  2. For each layer ℓ in `lens.layers`: `X = lens_vectors(model, lens, sample, ℓ)` → `(n, d)`, float32; subtract the column mean.
  3. `K = X @ X.T` → `(n, n)`; double-center: `K̃ = H K H`, `H = I − 11ᵀ/n`. Keep `K̃` in fp16 or on GPU; there are `L` of them at 100 MB each in fp32.
  4. `CKA[ℓ, ℓ′] = ⟨K̃_ℓ, K̃_ℓ′⟩_F / (‖K̃_ℓ‖_F ‖K̃_ℓ′‖_F)`, computed for all pairs in one `einsum` over flattened Grams.
- `figures.cka_heatmap(C, layers, path)` — square heatmap, layers on both axes, colorbar 0–1; save `C` as `.npy` beside it.
- Tests (`test_cka.py`, random lens on the stand-in): diagonal == 1 to 1e-5; symmetric; unchanged when every `X` is multiplied by the same random orthogonal matrix; on a random lens the off-diagonal is roughly uniform (report mean/std) so block structure on the real lens cannot be a plotting artifact.

### `io.py`
- `append_records(records, path)`, `write_manifest(run_dir, **fields)`, `upload_run(run_dir) -> str` (calls `hf upload orbitsoferis/jlens-specificity runs/<M>/ runs/<M>/ --repo-type dataset` via `subprocess`, returns commit URL), `sha256_of(path)`.

## Configs (write these at M0)

- `configs/tokens.yaml`:
  ```yaml
  language_tokens:
    es: ["Spanish", " Spanish", " spanish", "西班牙语"]
    fr: ["French",  " French",  " french",  "法语"]
  pairs:                       # form-matched; order = (es, fr)
    plain:  ["Spanish", "French"]
    space:  [" Spanish", " French"]
    lower:  [" spanish", " french"]
    zh:     ["西班牙语", "法语"]
  controls:
    # NOT hand-picked. Selected by scripts/m2_loading.py from the clean readout at question positions
    # (rule below), written to runs/M2/controls_proposed.yaml, approved by the human, then copied here.
    # control_ineligible applies ONLY to the selection of CONTROL tokens below. It has no effect on
    # treatment tokens, which come from `pairs` via pair_score. Its purpose: a control must be a
    # non-label edit, so no language-identity token may be chosen as a control target/pair member.
    control_ineligible:               # language-identity tokens, any script; human extends after seeing the M2 table
      ["Spanish"," Spanish"," spanish","French"," French"," french","西班牙语","法语","西班牙","法国",
       " español"," francés"," espagnol"," français"," langue"," idioma"," lengua"," bilingual"," translation"]
    label_to_present: {target: null}                   # filled from controls_proposed.yaml
    big_nonlabel:     {pair: null, norm_scale: 1.0}    # filled from controls_proposed.yaml
  answers:
    yes: ["Yes"]
    no:  ["No"]
    hello: {es: ["Hola"], fr: ["Bonjour"]}
  ```
  At M0 (stand-in tokenizer) and again at M1 (real tokenizer), assert which entries are single tokens; drop non-single-token entries **from the pairs list only**, and record what was dropped in `summary.md`.
- `configs/bands.yaml`: `workspace: null` (contiguous, [onset, motor onset)), `full: null` ([onset, last layer]), `early_late: null` (list of two [start, end] blocks). Human fills all three after reading M1's band-signature figure. M2 and M3 use `workspace` only and refuse to run while it is null; `full` and `early_late` are consumed by Stage 2 (a later prompt).

## Tests (must pass on CPU with the stand-in and a random lens)

- `test_env.py`: `require_env` raises on unset; never echoes.
- `test_lens.py`: `lens_vectors` shape; single-token assertion raises; final-layer readout with `J=I` equals logit lens.
- `test_prompts.py`: classes cover all tokens; spans reconstruct sentences from the templated text; `metric_pos` sits right after the prefill; every stimulus × question builds without flags on the stand-in tokenizer (or flags are listed).
- `test_interventions.py`:
  1. `swap` with `v_s == v_t` returns `h` bitwise.
  2. Orthogonal complement `(I − V pinv(V)) h` unchanged to 1e-5.
  3. With correlated `v_s, v_t` (cos≈0.7), `V @ c ≈ proj_V(h)`.
  4. `alpha` scales `h_new − h` linearly.
  5. `swap` ≠ steer (`h + alpha*v_t`) on random input.
  6. All-False mask → bitwise identity; masked-out positions bitwise unchanged.
  7. `apply` on the stand-in with an all-True mask reproduces the mini-paper's `run` to 1e-4 on logits.
  8. `random_direction` and `big_nonlabel` hit `norm_scale * target_norms` to 1e-3 per position; `big_nonlabel` leaves the label coordinate `c_label` unchanged to 1e-5; `label_to_present` produces a logged, finite `delta_h_norm` ratio to treatment.
- `test_loading.py`: `cos` in [−1, 1]; `rank` ≥ 1; `pair_score` on a synthetic frame gives the hand-computed value.
- `test_metrics.py`: `margin` on a hand-built logprob vector; `answer_ids` dedup; sign conventions; `flip` is True iff the answer-set argmax changed (test both directions and a no-change case).
- `test_io.py`: parquet round-trip; manifest fields present.

## Milestones (each ends in a commit + push to `stage01/replication`, an HF upload, and `runs/<M>/summary.md`; stop after each and wait for human sign-off)

**M0 — Scaffold (Phase A, Mac).** Everything above exists; `setup.sh` prints `SETUP OK (mac)`; all tests green on CPU; `scripts/m0_smoke.py` builds all 64 prompts on the stand-in, renders all 64 mask PNGs to `runs/M0/figures/masks/`, runs one `swap` cell and one `identity` cell on the stand-in with a random lens and writes a two-row parquet. `summary.md` includes: the exact templated prompt string for one stimulus, the tokens.yaml single-token table for the stand-in tokenizer, and any prompt flags.

**M1 — Lens validation (Phase B, GPU; first job on the box).**
0. `scripts/download.py` fetches the model at `configs/model.yaml` revision and the lens at `neuronpedia/jacobian-lens` file `qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt`, resolving the repo's current `main` sha via `HfApi().repo_info(...)` and recording it in `configs/lens.yaml`. Dump every non-tensor key of the `.pt` into `configs/lens.yaml`; record layers covered, dtype, one matrix's shape, `coverage_ratio`; copy `CREDIT.md` into `summary.md`; assert `lens.d_model == d_model(model)` else **stop**.
1. Final-layer agreement: top-10 overlap between J-lens and logit lens at 50 random positions from 5 arbitrary prompts. Report the distribution.
2. Readout reproduction: on `sp_01`, at 8 evenly spaced layers, top-5 readout at matrix positions; any `es` variant should be top-3 at mid layers. Emit the (position × layer) top-1 heatmap.
3. **Causal positive control**: Chinese antonym prompt from the paper (ask for the antonym of 小; expect 大). Swap ` big`→` long` and ` bigger`→` longer` (verify single-token; fall back to whichever variants are) at all content positions across layers 25–75% of depth; expect 长 to become top-1. Report clean and swapped top-5. If it fails at α=1, run α=2 and report both. **If both fail, stop; do not run M2.**
4. **Band signatures** (paper Fig. 27–28). Do not choose the band; compute these, plot them on one figure with shared x = layer, and the human fills `configs/bands.yaml`.
   a. `cka_matrix` over all lens layers (`n_tokens=5000`, `seed=0`); `cka_heatmap` + `.npy`. Expect a sharp early block and a possibly gradual top end (see the commentary's Qwen figure): CKA is used for the *onset*.
   b. **Next-token agreement by layer** (motor-onset detector): on the 64 stimulus prompts, at every content position, fraction where the lens top-1 (and top-5) equals the model's actual top-1 next token. Near zero in the workspace; rises steeply in the final layers. Report the curve and the first layer where top-1 agreement exceeds the midpoint between its median over layers 25–60% and its value at the last layer.
   c. **Excess kurtosis of the lens readout** by layer, same positions (near zero in sensory layers, rising at the workspace onset, falling at the top).
   d. Write `runs/M1/band_signatures.parquet` (layer, cka_onset_score, agreement_top1, agreement_top5, kurtosis) and `band_signatures.png`.
   Suggested rule for the human (not applied by code): `workspace = [CKA onset, motor onset)` where the onset is the **first** layer of the middle block (inclusive), not a layer above it; `full = [CKA onset, last layer]`; `early_late` = two blocks chosen from the signature figure (provisionally [18,30] and [39,47]).
5. Re-run the tokens.yaml single-token table on the real tokenizer.

**M2 — Workspace loading (Phase B; clean pass, no interventions).** For all 16 stimuli × 4 questions: `loading` for every token in `language_tokens` at every position and every layer in `bands.workspace`; clean `question_margin` and top-5 at `metric_pos`; accuracy table (report/hello/anomaly/content_probe correct?). Figures: per-stimulus loading heatmaps for the best es and best fr variant; `loading_summary_bars` over `bands.workspace`; `pair_score` table for all pairs. Log every prompt's `prompt_len`. Save `runs/M2/loadings.parquet`, `runs/M2/clean.parquet`.
**Control token selection (rule-based; human approves the output):** in `scripts/m2_loading.py`, after the clean pass:
1. Pool the top-25 readout at every `question`-class position over `bands.workspace`, across all 64 prompts. For each token: `freq` (fraction of prompts where it appears in top-25 at ≥1 question position), `mean_cos`, `mean_rank`.
2. Drop tokens in `controls.control_ineligible` (this list constrains control selection only; treatment tokens are unaffected), tokens that are not single tokens, and tokens with `freq < 0.5`. Also drop any token whose decoded string contains a language name (case-insensitive substring match on the control_ineligible list). Everything remaining is *eligible*.
3. `label_cos` = mean cos of the treatment target token at those positions (` French` on es-matrix prompts, ` Spanish` on fr-matrix, using pair A's members).
4. `label_to_present.target` = the eligible token with the highest `freq` among those with `mean_cos < label_cos`.
5. `big_nonlabel.pair` = the two eligible tokens with the highest `mean_cos`.
6. Write `runs/M2/controls_proposed.yaml` with the selections, the full eligible table, and the excluded table (so the human can see what the blocklist removed and what it missed). If step 4 or 5 returns nothing, write null and say so; the human decides.
The human reviews, may extend `controls.control_ineligible` and re-run step 2–6 (cheap: no forward passes), and copies the approved values into `configs/tokens.yaml`. **M3 refuses to run if `controls.label_to_present.target` or `controls.big_nonlabel.pair` is null.**

**M3 — Stage 1 grid (Phase B): the paper's protocol.** Position set = `{"question"}` only. Layers = `bands.workspace` **as filled by the human in `configs/bands.yaml` after M1**; the workspace band excludes the motor layers by definition (the motor band is a separate Stage 2 condition). α=1 for treatment.
Cells: 16 stimuli × 4 questions × direction {`m2i`: matrix-lang→intrusion-lang, `i2m`: reverse} × pair {the argmax of `pair_score` from M2} × kind {`swap`, `label_to_present`, `big_nonlabel`, `random_direction`, `identity`}. `random_direction` and `big_nonlabel` use `target_norms` = the matching treatment cell's per-position `delta_h_norm` (so the treatment cell runs first within each group); `label_to_present` runs at α=1 and reports its `delta_h_norm` ratio to treatment. Seed fixed and recorded.
Outputs: `runs/M3/records.parquet` (all `CellRecord` fields incl. `flip`, `top1_changed`); `panel_c` (flip rates + margins); `margin_vs_deltac` for `anomaly` and for `report`; ten randomly sampled raw cells per {question × direction × kind} in `summary.md` with clean and intervened top-5, verbatim.
Do not run other position sets, bands, α values, or pairs in M3.

## `summary.md` template (every milestone)
1. Environment (mac/cuda), commands, git commit, config hash, lens sha, model revision.
2. What passed by assertion; what was checked by eye; what was not checked.
3. Figures (embedded).
4. Anomalies, open questions, anything you were unsure about.
5. Artifact URL, parquet sha256s.
No scientific interpretation. Do not call a result "interesting." Report numbers.

## Ask before doing any of the following
Quantizing; editing anything under `stimuli/`; choosing or changing a band; changing a question string, a token list, or a control token; skipping or loosening a failing test; depending on the Neuronpedia website or API; running a position set other than question-only in M3; α ≠ 1 for treatment cells; using a stand-in that is not Qwen-family.

## HYGIENE INVARIANTS (binding)
1. **Lens before science:** validate the open-source Qwen J-Lens against a known positive (the Chinese antonym causal control) before any swap cell runs; a null from an unvalidated lens is attributed to the lens, not the phenomenon.
2. **Loading before swaps:** compute and save per-position workspace loading (cos(h, v_tok)) for source and target tokens in the clean pass before any intervention; log |Δc| for every swap cell.
3. **Interventions are located objects:** compute swap position indices per prompt from tokenizer offsets *after* chat-template application; classify every position as question / matrix / intrusion / template; never hardcode indices or reuse them across prompts; chat-template/special tokens are out of bounds unless deliberately targeted; skip the first ~4 high-norm positions.
4. **Render every mask:** for each run, log the full intervention specification (positions with class labels, layers, source/target tokens, coefficients, α, |Δc|) and emit a sanity figure of the prompt with intervened tokens highlighted, saved alongside the results it belongs to.
5. **Swap math is the paper's, exactly:** V=[v_s,v_t], c=V†h, h←h+V(σ(c)−c), orthogonal complement untouched; unit-test it (norm preserved up to the swapped component; zero effect when v_s=v_t; orthogonal component bitwise unchanged).
6. **Metrics are logprob margins**, never sampled binary answers; log full next-token distributions for every condition. A binary "No" is not evidence of a semantic flip.
7. **No treatment cell without its controls in the same run:** label→present-non-label swap, large non-label J-space swap (‖ΔH‖ ≥ treatment), matched-norm random direction, and the content probe, always; positive-control conditions (report/hello) run before any anomaly cell — if positives don't flip, stop and report, do not proceed.
8. **Layer bands come from the project's own CKA plot**, not Claude's boundaries and not any external default. The `workspace` band excludes the motor layers by definition; `full` and `early_late` are separate conditions in a later stage.
9. **Conduct:** never mock data, fabricate outputs, hardcode results, or delete/weaken failing tests; on any failure or ambiguity, stop and report rather than improvising; every milestone ends in an artifact (plot/table/assert) for human inspection, and no next stage begins until the human has signed off.
10. **Sync artifacts off the ephemeral instance as they are produced** — masks, logits, loadings, configs, seeds, exact tokenized prompt strings — and report the sync destination in the milestone summary.
