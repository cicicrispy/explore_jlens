"""M2 (Phase B; clean pass, no interventions): workspace loading for all 64 prompts, saved per
prompt, at EVERY lens layer.

Every run gets its own folder, runs/M2/<experiment name>_<UTC start time>/ (settings copy, see
jlens_spec/runs.py). For each prompt, ONE forward pass saves (file name <stimulus>_<question>.parquet
in each folder):
    loadings/  cos(h, v_token) and full-readout rank for all 8 language tokens, every position x layer
    topk/      the lens readout's top-`save_topk` tokens at every position x layer (M3's control pool)
    answer/    the model's answer: full-vocabulary logprobs (fp16), top-`save_topk`, margin, correct?
    tokens/    the prompt's tokens and their position classes
Each prompt's files are uploaded as soon as they are written. Stopping and starting again RESUMES the
run (see runs.start_or_resume): prompts whose files are in the store are skipped.

At the end, from the saved files only: pair scores for every pair over the workspace band (and which
wins -- M3's treatment pair), the accuracy table, figures (never uploaded), summary.md last.

Refuses to start unless configs/bands.yaml is complete and consistent (env.check_bands) and, for a
real run, the M1 validation run named in the experiment file is finished and passed.

    python scripts/m2_loading.py                     # configs/experiments/m2_loading.yaml; resumes if unfinished
    python scripts/m2_loading.py --fresh             # always a new run
    python scripts/m2_loading.py --resume runs/M2/loading_20260912-031000
    python scripts/m2_loading.py --experiment configs/experiments/dryrun/m2_loading.yaml   # Mac dry run
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
import pyarrow as pa
import torch
import yaml
from tqdm import tqdm

from jlens_spec import env

env.bootstrap()

from jlens_spec import figures  # noqa: E402
from jlens_spec import io as io_mod  # noqa: E402
from jlens_spec import loading as loading_mod  # noqa: E402
from jlens_spec import metrics  # noqa: E402
from jlens_spec import pipeline  # noqa: E402
from jlens_spec import prompts as prompts_mod  # noqa: E402
from jlens_spec import runs as runs_mod  # noqa: E402

DEFAULT_EXPERIMENT = "configs/experiments/m2_loading.yaml"
FOLDERS = ("loadings", "topk", "answer", "tokens")


def _expected_label(question_key: str, stimulus: dict) -> str:
    """Ground truth for the accuracy table, derived from stimuli.json (every passage here has
    exactly one intrusion sentence, so `anomaly`'s correct answer is always 'yes'; `report`/`hello`
    should reflect the matrix language, i.e. metrics.argmax_label's 'source'; `content` reflects
    the passage's `outdoors` field)."""
    if question_key in ("report", "hello"):
        return "source"
    if question_key == "anomaly":
        return "yes"
    if question_key == "content":
        return "yes" if stimulus["outdoors"] else "no"
    raise ValueError(f"unknown question_key {question_key!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help=f"experiment file (default {DEFAULT_EXPERIMENT})")
    ap.add_argument("--fresh", action="store_true", help="start a new run even if the latest one is unfinished")
    ap.add_argument("--resume", default=None, help="resume this unfinished run folder")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu",
                    help="dry runs only: mps or cpu (a real run uses the GPU via device_map='auto')")
    args = ap.parse_args()
    git_commit = env.git_commit()  # the code this process loaded; every row of this launch carries it

    exp0 = pipeline.read_experiment(args.experiment)
    dry = pipeline.is_dryrun(exp0)
    store = pipeline.store_for(exp0)
    if not dry:
        env.require_env("HF_TOKEN")
    # Checks that need no model, before any run folder is created (repeated on the run's own
    # settings copy, which is what a resumed run uses).
    pipeline.check_m1_passed(exp0, store)
    with open(pipeline.settings_files(exp0)[-1]) as f:
        env.check_bands(yaml.safe_load(f))

    run, resumed = runs_mod.start_or_resume("M2", args.experiment, pipeline.settings_files(exp0), store,
                                            root=pipeline.runs_root(exp0), fresh=args.fresh, resume=args.resume)
    exp = run.experiment
    m1_line = pipeline.check_m1_passed(exp, store)
    model_cfg, lens_cfg, tokens_raw = run.load("model.yaml"), run.load("lens.yaml"), run.load("tokens.yaml")
    stim, prompt_format, bands = run.load("stimuli.json"), run.load("prompt_format.yaml"), run.load("bands.yaml")
    save_k = exp["save_topk"]

    print(f"[1/4] Loading the model and the lens ({pipeline.environment(exp)}) ...", flush=True)
    model, lens = pipeline.load_model_and_lens(exp, model_cfg, lens_cfg, args.device)
    band_layers = env.check_bands(bands, lens.layers)
    workspace = band_layers["workspace"]
    tok = model.tokenizer
    tokens_cfg = metrics.build_tokens_cfg(tok, tokens_raw)
    fmt = prompts_mod.make_fmt(tok, stim, prompt_format)
    lang_tokens = sorted({t for forms in tokens_raw["language_tokens"].values() for t in forms})
    lang_ids = [tok.encode(t, add_special_tokens=False)[0] for t in lang_tokens
                if len(tok.encode(t, add_special_tokens=False)) == 1]
    pairs, dropped_pairs = loading_mod.single_token_pairs(tok, tokens_raw["pairs"])
    assert pairs, f"no pair in tokens.yaml is single-token under this tokenizer (dropped: {dropped_pairs})"

    jobs = [(s, q) for s in stim["passages"] for q in stim["questions"]]
    done = runs_mod.uploaded_files(run.dir, store)
    todo = [(s, q) for s, q in jobs if not all(f"{d}/{s['id']}_{q}.parquet" in done for d in FOLDERS)]
    print(f"[2/4] Clean pass: {len(jobs) - len(todo)} of {len(jobs)} prompts already done (in the {store.name}); "
          f"{len(todo)} to run, every one of the {len(lens.layers)} lens layers ...", flush=True)
    uploader = io_mod.BackgroundUploader(run.dir, store=store)
    flags = {}
    for stimulus, q in tqdm(todo, desc="prompts", unit="prompt"):
        key = f"{stimulus['id']}_{q}"
        p = prompts_mod.build_prompt(stimulus, q, fmt)
        if p.flags:
            flags[key] = p.flags
        loadings, topk, answer_logits = loading_mod.prompt_pass(model, lens, p, lang_ids, save_k)
        logprobs = torch.log_softmax(answer_logits.float(), dim=-1)
        label = metrics.argmax_label(q, logprobs, tokens_cfg, stimulus["matrix_lang"])
        expected = _expected_label(q, stimulus)
        answer = {
            "stimulus_id": stimulus["id"], "question_key": q, "matrix_lang": stimulus["matrix_lang"],
            "prompt_len": len(p.input_ids), "metric_pos": p.metric_pos,
            "margin": metrics.question_margin(q, logprobs, tokens_cfg, stimulus["matrix_lang"]),
            "answer_label": label, "expected_label": expected, "correct": label == expected,
            # flags as one string: an empty list would give some files a different column type
            "topk": metrics.topk_tokens(answer_logits, tok, save_k), "flags": "; ".join(p.flags),
            "run_id": run.run_id, "git_commit": git_commit,
            "logprobs_fp16": logprobs.to(torch.float16).cpu().numpy(),
        }
        stamp = {"run_id": run.run_id, "git_commit": git_commit}
        io_mod.write_parquet(loadings.assign(**stamp), run.dir / "loadings" / f"{key}.parquet")
        n = topk.num_rows
        io_mod.write_parquet(topk.append_column("run_id", pa.array([run.run_id] * n))
                             .append_column("git_commit", pa.array([git_commit] * n)),
                             run.dir / "topk" / f"{key}.parquet")
        io_mod.write_parquet(io_mod.records_table([answer]), run.dir / "answer" / f"{key}.parquet")
        io_mod.write_parquet([{**r, **stamp} for r in prompts_mod.mask_rows(p, torch.zeros(len(p.input_ids), dtype=torch.bool))],
                             run.dir / "tokens" / f"{key}.parquet")
        uploader.trigger(files=[f"{d}/{key}.parquet" for d in FOLDERS])
    _, bg_error = uploader.flush()
    uploader.shutdown()
    downloaded = runs_mod.sync_down(run.dir, store)

    print("[3/4] Pair scores, accuracy and figures, from the saved files ...", flush=True)
    answers = pd.read_parquet(run.dir / "answer", columns=["stimulus_id", "question_key", "matrix_lang", "margin",
                                                           "answer_label", "expected_label", "correct",
                                                           "prompt_len", "git_commit"])
    assert len(answers) == len(jobs), f"expected {len(jobs)} answer rows, found {len(answers)}"
    ws = pd.read_parquet(run.dir / "loadings", filters=[("layer", "in", workspace)],
                         columns=["stimulus_id", "question_key", "pos", "class", "layer", "token", "cos"])
    pair_scores = {name: loading_mod.pair_score(ws, tuple(pair), workspace) for name, pair in pairs.items()}
    nan_pairs = sorted(n for n, s in pair_scores.items() if s != s)
    valid = {n: s for n, s in pair_scores.items() if s == s}
    assert valid, f"every pair_score is NaN ({nan_pairs}) -- loading rows don't match pair tokens"
    best_pair = max(valid, key=valid.get)
    with open(run.dir / "pair_scores.yaml", "w") as f:
        yaml.safe_dump({"band": "workspace", "band_layers": workspace,
                        "pair_scores": {n: (None if s != s else float(s)) for n, s in pair_scores.items()},
                        "pair_score_argmax": best_pair, "pair_words": pairs[best_pair],
                        "pairs_nan_score": nan_pairs, "pairs_dropped_not_single_token": dropped_pairs},
                       f, sort_keys=False, allow_unicode=True)
    (run.dir / "figure_params.json").write_text(json.dumps({"band": workspace, "band_name": "workspace",
                                                            "pairs": pairs}, indent=2, ensure_ascii=False))
    written = figures.make_figures(run.dir)
    accuracy = answers.groupby("question_key")["correct"].agg(["mean", "sum", "count"])

    print("[4/4] Checksums, summary, upload ...", flush=True)
    data_files = sorted(p for d in FOLDERS for p in (run.dir / d).glob("*.parquet"))
    per_folder = ", ".join(f"{d}/ x{sum(p.parent.name == d for p in data_files)}" for d in FOLDERS)
    checksums = "\n".join(f"{io_mod.sha256_of(p)}  {p.relative_to(run.dir).as_posix()}" for p in data_files)
    (run.dir / "checksums.txt").write_text(checksums + "\n")
    commits = sorted(answers["git_commit"].unique())
    summary_lines = [
        f"# M2 summary -- run {run.run_id}", "",
        "## 1. Environment",
        f"- Environment: {pipeline.environment(exp)}",
        f"- git commit(s) that produced the rows: {', '.join(commits)}"
        + (" -- more than one: the run was resumed with --resume after a code change" if len(commits) > 1 else ""),
        f"- resumed: {resumed}; experiment file: {args.experiment}; config_hash (every file in settings/): "
        f"{run.config_hash()}",
        f"- model: {model_cfg['standin_hf_id'] + ' (stand-in)' if dry else model_cfg['hf_id'] + ' @ ' + model_cfg['revision']}",
        f"- lens: {'random lens, seed ' + str(exp['dryrun']['lens_seed']) if dry else lens_cfg['repo'] + ' @ ' + lens_cfg['revision_sha']}",
        f"- {m1_line}",
        f"- bands (settings/bands.yaml): workspace {bands['workspace']}, full {bands['full']}, "
        f"early_late {bands['early_late']} -- complete and consistent (env.check_bands)",
        f"- saved at every one of the {len(lens.layers)} lens layers ({lens.layers[0]}..{lens.layers[-1]}); "
        f"top-{save_k} per position; dataset: {store.name}", "",
        "## 2. What passed by assertion / checked by eye / not checked",
        f"- {len(answers)}/{len(jobs)} prompts, one clean forward pass each; prompts with build flags: "
        f"{flags or 'none'}.",
        "- Accuracy (answer-set argmax vs the passage's ground truth; margins are logprob margins):",
        "  | question | correct | of | rate |", "  |---|---|---|---|",
        *[f"  | {q} | {int(r['sum'])} | {int(r['count'])} | {r['mean']:.3f} |" for q, r in accuracy.iterrows()],
        f"- pair_score over the workspace layers {figures.layers_text(workspace)}: "
        + ", ".join(f"{n} = {s:.4f}" if s == s else f"{n} = NaN" for n, s in pair_scores.items())
        + f"; argmax = {best_pair!r} {pairs[best_pair]} -- M3's treatment pair (pair_scores.yaml).",
        f"- Pairs dropped (not single tokens): {dropped_pairs or 'none'}.",
        "- Control tokens are NOT picked here: M3 picks them automatically at the start of each positives "
        "run, from this run's topk/ files.",
        "- NOT checked: any scientific reading of these numbers.", "",
        "## 3. Figures",
        f"- {len(written)} figure files in {run.dir}/figures/png/: per prompt, loadings/ (cos(h, v_token)) and "
        "ranks/ (the token's rank in the lens readout; white line = rank 100) -- every pair's Spanish and French "
        "member, every layer, workspace layers dashed, one color scale per kind for the whole run; plus "
        "loading_summary_bars (all prompts / Spanish passages / French passages). M2 figures are NOT uploaded; "
        f"redraw from the data anywhere, without the model: `python scripts/make_figures.py {run.dir} --format pdf`", "",
        "## 4. Anomalies / open questions",
        f"- Background uploads: {uploader.n_uploads} ok, {uploader.n_failures} failed"
        + (f"; last error: {bg_error}" if bg_error else "") + (f"; still queued: {sorted(uploader.pending())}"
                                                                if uploader.pending() else ""),
        f"- Files downloaded from the store at the end (computed on another machine): {len(downloaded)}", "",
        "## 5. Checksums",
        f"- {len(data_files)} data files ({per_folder}); sha256 of each in checksums.txt "
        f"(sha256 {io_mod.sha256_of(run.dir / 'checksums.txt')})",
    ]
    url = io_mod.finalize_run(run.dir, summary_lines, store=store)
    if url is None:
        print(f"M2 computed everything, but the upload FAILED, so this run counts as unfinished "
              f"(see {run.dir}/upload_error.txt). Run the same command again to retry.")
    else:
        print(f"M2 complete. See {run.dir}/summary.md")


if __name__ == "__main__":
    main()
