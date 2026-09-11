"""M3 (Phase B): the paper's protocol -- Stage 1 grid. One run = one experiment file: some questions
x one position set x one band, alpha = 1 for treatment. The planned runs (configs/experiments/):

    m3_positives_question.yaml   report + hello,      edits the question sentence
    m3_anomaly_question.yaml     anomaly + content,   edits the question sentence
    m3_positives_message.yaml    report + hello,      edits the whole user message
    m3_anomaly_message.yaml      anomaly + content,   edits the whole user message

Invariant 7's "positive controls first; if they don't flip, stop": run a position set's POSITIVES
run first and read its summary; only then launch the ANOMALY run (it refuses unless its positives
run is finished and used the same M2 run, band and position set). The decision is yours -- no code
applies a threshold.

The control tokens are picked BEFORE the positives run, by a controls run (scripts/m3_controls.py),
which stops so you can read them. The positives run names the controls run you accepted
(`controls_run:`), refuses it unless it is finished and matches this run (controls.selection_problems
-- the code version is not compared), and copies its selection.yaml into controls/; the anomaly run
reuses that copy. Then, for every prompt of the run's questions and both directions: identity, swap
(the treatment), random_direction, and 3 label_to_present + 3 big_nonlabel cells (one per control) =
18 cells per prompt. Per prompt, in files named <stimulus>_<question>.parquet:
    records/   one row per cell (margin, flip, top-100 next tokens, full-vocab logprobs fp16, logs)
    details/   per cell x band layer x position: where the stream ACTUALLY changed + the planned log
    tokens/    the prompt's tokens, classes, and which positions the position set plans to edit
Each prompt's files are uploaded as soon as they are written. Running the same command again
RESUMES an unfinished run (only if the code is unchanged -- see runs.start_or_resume). An edit that
lands outside its planned positions stops the run (hard failure). Figures and summary.md come last,
from the saved files only; the summary figures are uploaded with the run, the mask figures never.

    python scripts/m3_controls.py --experiment configs/experiments/m3_controls_question.yaml
    python scripts/m3_grid.py --experiment configs/experiments/m3_positives_question.yaml
    python scripts/m3_grid.py --experiment configs/experiments/m3_anomaly_question.yaml
    python scripts/m3_grid.py --experiment ... --fresh | --resume runs/M3/<run folder>
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml
from tqdm import tqdm

from jlens_spec import env

env.bootstrap()

from jlens_spec import controls as controls_mod  # noqa: E402
from jlens_spec import figures  # noqa: E402
from jlens_spec import io as io_mod  # noqa: E402
from jlens_spec import metrics  # noqa: E402
from jlens_spec import pipeline  # noqa: E402
from jlens_spec import prompts as prompts_mod  # noqa: E402
from jlens_spec import runner  # noqa: E402
from jlens_spec import runs as runs_mod  # noqa: E402

FOLDERS = ("records", "details", "tokens")
SELECTION = "controls/selection.yaml"   # a positives run's copy of its controls run's selection


def _run_checks(selection, questions) -> list[str]:
    """The checks (row x question) whose cells this run makes."""
    return [k for k, t in selection["checks"].items() if t["question"] in questions]


def _cells(exp, stim, pair_name, band_layers, selection) -> list:
    classes = set(prompts_mod.POSITION_SETS[exp["position_set"]])
    base = dict(pair_name=pair_name, layers=band_layers, position_set=classes, alpha=exp["alpha"], seed=exp["seed"])
    cells = []
    for s in stim["passages"]:
        for q in exp["questions"]:
            for d in exp["directions"]:
                common = dict(stimulus_id=s["id"], question_key=q, direction=d, **base)
                cells += [runner.Cell(kind=k, **common) for k in ("identity", "swap", "random_direction")]
                check = f"{q}_{s['matrix_lang']}_{d}"          # this cell's own label_to_present controls
                for ctl in selection["label_to_present"][check]:
                    cells.append(runner.Cell(kind="label_to_present", control_index=ctl["control_index"],
                                             control_tokens=[ctl["token_id"]], control_text=[ctl["token"]],
                                             control_tier=ctl["tier"], **common))
                for ctl in selection["big_nonlabel"]:
                    cells.append(runner.Cell(kind="big_nonlabel", control_index=ctl["control_index"],
                                             control_tokens=[ctl["a_id"], ctl["b_id"]],
                                             control_text=[ctl["a_token"], ctl["b_token"]],
                                             control_tier=ctl["tier"], **common))
    return cells


def _logprob_matrix(run_dir, kinds) -> tuple[pd.DataFrame, np.ndarray]:
    """The stored 16-bit log-probabilities of every cell of these kinds (rows aligned with the table)."""
    t = pq.read_table(Path(run_dir) / "records", filters=[("kind", "in", list(kinds))],
                      columns=["stimulus_id", "question_key", "direction", "kind", "control_index", "logprobs_fp16"])
    col = t.column("logprobs_fp16").combine_chunks()
    lp = col.flatten().to_numpy(zero_copy_only=False).reshape(len(t), col.type.list_size).astype(np.float16)
    return t.drop_columns(["logprobs_fp16"]).to_pandas(), lp


def _direction_checks(run_dir) -> list[str]:
    """Identity cells of the two directions should be identical; swap / random_direction /
    big_nonlabel are the same edit in both directions (the swap is symmetric), so they should match
    up to rounding. The log-probabilities are stored in 16-bit, so two values a hair apart can land
    on neighbouring 16-bit steps (0.0005 near -0.5, 0.03 near -40): "within one 16-bit step" is the
    storage precision. Reported, not failed."""
    meta, lp = _logprob_matrix(run_dir, ["identity", "swap", "random_direction", "big_nonlabel"])
    lines = []
    for kind in ("identity", "swap", "random_direction", "big_nonlabel"):
        sub = meta[meta["kind"] == kind]
        diffs, unequal, within = [], 0, 0
        for _, g in sub.groupby(["stimulus_id", "question_key", "control_index"]):
            if set(g["direction"]) != {"m2i", "i2m"}:
                continue
            a, b = lp[g.index[g["direction"] == "m2i"][0]], lp[g.index[g["direction"] == "i2m"][0]]
            d = np.abs(a.astype(np.float32) - b.astype(np.float32))
            step = np.spacing(np.maximum(np.abs(a), np.abs(b))).astype(np.float32)
            diffs.append(float(d.max()))
            unequal += int(not np.array_equal(a, b))
            within += int((d <= step).all())
        if diffs:
            lines.append(f"  - {kind}: {len(diffs)} m2i/i2m cell pairs; bitwise identical: {len(diffs) - unequal}; "
                         f"every value within one 16-bit step: {within}/{len(diffs)}; "
                         f"max |Δ logprob|: {max(diffs):.3g}")
    return lines


def _size_checks(run_dir, lang_of: dict) -> tuple[pd.DataFrame, list[str]]:
    """Each control's measured edit size vs the treatment's (same prompt and direction): mean over
    planned positions x layers of |Δc| and of ||Δh||, as ratios control / swap. label_to_present is
    reported per check (each check has its own tokens: its 3 controls pooled), the others per
    control. `lang_of` = {stimulus_id: passage language}."""
    d = pd.read_parquet(Path(run_dir) / "details", columns=["stimulus_id", "question_key", "direction", "kind",
                                                            "control_index", "planned", "delta_c_norm", "delta_h_norm"])
    d = d[d["planned"]]
    cell = d.groupby(["stimulus_id", "question_key", "direction", "kind", "control_index"])[
        ["delta_c_norm", "delta_h_norm"]].mean().reset_index()
    swap = cell[cell["kind"] == "swap"].set_index(["stimulus_id", "question_key", "direction"])
    ctl = cell[cell["kind"].isin(["label_to_present", "big_nonlabel", "random_direction"])].copy()
    idx = list(zip(ctl["stimulus_id"], ctl["question_key"], ctl["direction"]))
    ctl["dc_ratio"] = ctl["delta_c_norm"].to_numpy() / swap.loc[idx, "delta_c_norm"].to_numpy()
    ctl["dh_ratio"] = ctl["delta_h_norm"].to_numpy() / swap.loc[idx, "delta_h_norm"].to_numpy()
    ltp = ctl["kind"] == "label_to_present"
    ctl["group"] = np.where(ltp, "label_to_present, " + ctl["question_key"] + "_" + ctl["stimulus_id"].map(lang_of)
                            + "_" + ctl["direction"] + " (its controls pooled)",
                            ctl["kind"] + "[" + ctl["control_index"].astype(str) + "]")
    lines = []
    for group, g in ctl.groupby("group", sort=True):
        dc = "n/a (no swap coordinates)" if g["kind"].iloc[0] == "random_direction" else \
            f"mean {g['dc_ratio'].mean():.3f}, min {g['dc_ratio'].min():.3f}, cells below 1: {int((g['dc_ratio'] < 1).sum())}/{len(g)}"
        lines.append(f"  - {group}: |Δc| ratio {dc}; ||Δh|| ratio mean {g['dh_ratio'].mean():.3f}, "
                     f"min {g['dh_ratio'].min():.3f}")
    return ctl, lines


def _check_experiment(exp, path, store, run=None, resumed=False):
    """Refuse bad references before any work: the M1 run passed (real runs), the M2 run and -- for a
    positives run -- the controls run, or -- for an anomaly run -- the positives run exist and are
    finished. Returns (positives?, the M1 summary line, M2 run name, controls run name or None,
    positives run name or None). With `run`, the resolved names are saved in the run's refs.yaml the
    first time and read back on resume, so a resumed run keeps pointing at the runs it started with
    (a dry run's "latest:" could otherwise pick a newer one)."""
    if "controls" in exp:
        raise SystemExit(f"{path} has a `controls:` block: the controls are picked by scripts/m3_controls.py now "
                         "(its own experiment file, e.g. m3_controls_question.yaml); a positives run names the "
                         "controls run you accepted in `controls_run:`")
    positives = "positives_run" not in exp
    if positives != ("controls_run" in exp):
        raise SystemExit(f"{path}: a positives run has `controls_run:`, an anomaly run `positives_run:` "
                         "-- exactly one of the two")
    if exp["position_set"] not in prompts_mod.POSITION_SETS:
        raise SystemExit(f"position_set must be one of {sorted(prompts_mod.POSITION_SETS)}")
    m1_line = pipeline.check_m1_passed(exp, store)
    if not exp.get("from_m2_run"):
        raise SystemExit("set from_m2_run in the experiment file (the finished M2 run, runs/M2/<this>)")
    if positives and not exp.get("controls_run"):
        raise SystemExit("set controls_run in the experiment file (the scripts/m3_controls.py run of this position "
                         "set whose picks you accepted, runs/M3/<this>)")
    if not positives and not exp.get("positives_run"):
        raise SystemExit("set positives_run in the experiment file (this position set's finished positives run)")
    refs_path = run.dir / "refs.yaml" if run is not None else None
    if refs_path is not None and resumed and (refs_path.exists() or store.exists(refs_path)):
        refs = yaml.safe_load(pipeline.fetch(refs_path, store).read_text())
    else:
        refs = {"from_m2_run": pipeline.resolve_run(exp["from_m2_run"], "M2", exp, store),
                "controls_run": pipeline.resolve_run(exp["controls_run"], "M3", exp, store) if positives else None,
                "positives_run": None if positives else pipeline.resolve_run(exp["positives_run"], "M3", exp, store)}
        if refs_path is not None:
            refs_path.write_text(yaml.safe_dump(refs))
    root = pipeline.runs_root(exp)
    pipeline.require_finished(root / "M2" / refs["from_m2_run"], store, "M2 run")
    if refs["controls_run"]:
        pipeline.require_finished(root / "M3" / refs["controls_run"], store, "controls run")
    if refs["positives_run"]:
        pipeline.require_finished(root / "M3" / refs["positives_run"], store, "positives run")
    return positives, m1_line, refs["from_m2_run"], refs["controls_run"], refs["positives_run"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, help="an M3 experiment file, e.g. configs/experiments/m3_positives_question.yaml")
    ap.add_argument("--fresh", action="store_true", help="start a new run even if the latest one is unfinished")
    ap.add_argument("--resume", default=None, help="resume this unfinished run folder")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu",
                    help="dry runs only: mps or cpu (a real run uses the GPU via device_map='auto')")
    args = ap.parse_args()
    git_commit = env.git_commit()  # the code this process loaded; every row of this launch carries it

    exp0 = pipeline.read_experiment(args.experiment)
    dry, store, root = pipeline.is_dryrun(exp0), pipeline.store_for(exp0), pipeline.runs_root(exp0)
    if not dry:
        env.require_env("HF_TOKEN")
    # Checks that need no model, before any run folder is created (repeated below on the run's own
    # settings copy, which is what a resumed run uses).
    _check_experiment(exp0, args.experiment, store)
    with open(pipeline.settings_files(exp0)[-1]) as f:
        env.check_bands(yaml.safe_load(f))

    run, resumed = runs_mod.start_or_resume("M3", args.experiment, pipeline.settings_files(exp0), store,
                                            root=root, fresh=args.fresh, resume=args.resume)
    exp = run.experiment
    positives, m1_line, m2_run, ctl_run, pos_run = _check_experiment(exp, args.experiment, store, run, resumed)
    m2_dir = root / "M2" / m2_run
    model_cfg, lens_cfg, tokens_raw = run.load("model.yaml"), run.load("lens.yaml"), run.load("tokens.yaml")
    stim, prompt_format, bands = run.load("stimuli.json"), run.load("prompt_format.yaml"), run.load("bands.yaml")
    m2 = yaml.safe_load(pipeline.fetch(m2_dir / "pair_scores.yaml", store).read_text())
    pair_name, pair_words = m2["pair_score_argmax"], m2["pair_words"]
    if tokens_raw["pairs"].get(pair_name) != pair_words:
        raise SystemExit(f"M2's winning pair {pair_name} = {pair_words} is not in this run's tokens.yaml pairs")

    print(f"[1/5] Loading the model and the lens ({pipeline.environment(exp)}) ...", flush=True)
    model, lens = pipeline.load_model_and_lens(exp, model_cfg, lens_cfg, args.device)
    band_layers = env.check_bands(bands, lens.layers)[exp["band"]]
    tok = model.tokenizer
    fmt = prompts_mod.make_fmt(tok, stim, prompt_format)

    print("[2/5] Control tokens ...", flush=True)
    # A positives run takes the selection of the controls run it names (resumed or not: refs.yaml pins
    # that run) and keeps a copy in its own controls/; an anomaly run takes its positives run's copy.
    # Either must match this run's settings; the code version is not compared (selection_problems).
    source = ctl_run if positives else pos_run
    selection = yaml.safe_load(pipeline.fetch(root / "M3" / source / SELECTION, store).read_text())
    problems = controls_mod.selection_problems(
        selection, position_set=exp["position_set"], band_layers=band_layers, from_m2_run=m2_run,
        pair_words=pair_words, skip_first=exp["skip_first"])
    if problems:
        raise SystemExit(f"{'controls' if positives else 'positives'} run {source} can't be used by this run: "
                         + "; ".join(problems))
    if positives:
        selection = {**selection, "controls_run": ctl_run}
        (run.dir / "controls").mkdir(exist_ok=True)
        with open(run.dir / SELECTION, "w") as f:
            yaml.safe_dump(selection, f, sort_keys=False, allow_unicode=True)
        controls_source = f"from controls run {ctl_run} (scripts/m3_controls.py), named in the experiment file"
    else:
        controls_source = (f"reused from positives run {pos_run}, which took them from controls run "
                           f"{selection.get('controls_run')}")
    ctl_lines = controls_mod.control_lines(selection, exp["questions"])
    print("\n".join(ctl_lines), flush=True)

    cells = _cells(exp, stim, pair_name, band_layers, selection)
    groups = runner.group_by_prompt(cells)
    done = runs_mod.uploaded_files(run.dir, store)
    todo = [k for k in groups if not all(f"{d}/{k[0]}_{k[1]}.parquet" in done for d in FOLDERS)]
    cfgs = {
        "stimuli": {s["id"]: s for s in stim["passages"]}, "fmt": fmt, "tokens_raw": tokens_raw,
        "tokens_cfg": metrics.build_tokens_cfg(tok, tokens_raw), "skip_first": exp["skip_first"],
        "save_topk": exp["save_topk"], "norm_scale": selection["norm_scale"], "config_hash": run.config_hash(),
        "lens_sha": f"random-lens-seed-{exp['dryrun']['lens_seed']}" if dry else lens_cfg["revision_sha"],
        "model_revision": "standin" if dry else model_cfg["revision"], "run_id": run.run_id,
        "git_commit": git_commit,
    }
    uploader = io_mod.BackgroundUploader(run.dir, store=store)
    uploader.trigger(files=[f for f in ("refs.yaml", SELECTION) if (run.dir / f).exists()])
    print(f"[3/5] Cells: {len(groups) - len(todo)} of {len(groups)} prompts already done (in the {store.name}); "
          f"{len(todo)} to run, {len(cells) // len(groups)} cells each, layers {figures.layers_text(band_layers)} "
          f"({exp['band']}), position set '{exp['position_set']}' ...", flush=True)
    target_norms_cache = {}
    for key in tqdm(todo, desc="prompts", unit="prompt"):
        name = f"{key[0]}_{key[1]}"
        records, details = runner.run_prompt(model, lens, groups[key], cfgs, target_norms_cache)
        p = prompts_mod.build_prompt(cfgs["stimuli"][key[0]], key[1], fmt)
        m = prompts_mod.mask(p, set(prompts_mod.POSITION_SETS[exp["position_set"]]), skip_first=exp["skip_first"])
        io_mod.write_parquet(io_mod.records_table(records), run.dir / "records" / f"{name}.parquet")
        io_mod.write_parquet(details, run.dir / "details" / f"{name}.parquet")
        io_mod.write_parquet([{**r, "run_id": run.run_id, "git_commit": git_commit} for r in prompts_mod.mask_rows(p, m)],
                             run.dir / "tokens" / f"{name}.parquet")
        uploader.trigger(files=[f"{d}/{name}.parquet" for d in FOLDERS])
    _, bg_error = uploader.flush()
    uploader.shutdown()
    downloaded = runs_mod.sync_down(run.dir, store)

    print("[4/5] Checks and figures, from the saved files ...", flush=True)
    rec = figures.read_records(run.dir, ["stimulus_id", "question_key", "direction", "kind", "control_index",
                                         "control_text", "control_tier", "flip", "margin", "clean_margin",
                                         "top1_changed", "topk", "n_planned", "n_planned_unchanged", "git_commit"])
    expected_n = len(cells)
    per_group = rec.groupby(["stimulus_id", "question_key", "direction"])["kind"].apply(sorted)
    lang_of = {s["id"]: s["matrix_lang"] for s in stim["passages"]}

    def want(sid, q, d) -> list[str]:  # the kinds one prompt x direction must have (its check's controls)
        return sorted(["identity", "swap", "random_direction"] + ["big_nonlabel"] * len(selection["big_nonlabel"])
                      + ["label_to_present"] * len(selection["label_to_present"][f"{q}_{lang_of[sid]}_{d}"]))

    incomplete = [k for k, v in per_group.items() if v != want(*k)]
    cells_per_group = len(cells) // max(len(stim["passages"]) * len(exp["questions"]) * len(exp["directions"]), 1)
    direction_lines = _direction_checks(run.dir)
    _, size_lines = _size_checks(run.dir, lang_of)
    (run.dir / "figure_params.json").write_text(json.dumps({
        "band": exp["band"], "band_layers": band_layers, "position_set": exp["position_set"],
        "pair_name": pair_name, "pair_words": pair_words, "questions": exp["questions"]}, indent=2, ensure_ascii=False))
    written = figures.make_figures(run.dir)

    ctl = rec[rec["kind"].isin(["label_to_present", "big_nonlabel"])]
    per_control = ctl.assign(control=[" <-> ".join(t) for t in ctl["control_text"]]).groupby(
        ["kind", "control_index", "control", "question_key", "direction"]).agg(
        flip_rate=("flip", lambda s: float(s.dropna().astype(bool).mean())), mean_margin=("margin", "mean"),
        n=("margin", "size")).reset_index()
    rng = random.Random(exp["seed"])
    sampled = ["## 6. Ten randomly sampled raw cells per {question x direction x kind} (clean and intervened top-5)", ""]
    for (q, d, kind), g in rec.groupby(["question_key", "direction", "kind"]):
        clean = rec[(rec["kind"] == "identity") & (rec["question_key"] == q) & (rec["direction"] == d)] \
            .set_index("stimulus_id")["topk"]
        for i in sorted(rng.sample(range(len(g)), min(10, len(g)))):
            r = g.iloc[i]
            top5 = lambda t: [(x["token"], round(float(x["logprob"]), 3)) for x in list(t)[:5]]  # noqa: E731
            sampled.append(f"- {q} / {d} / {kind}"
                           + (f"[{r['control_index']}] {list(r['control_text'])}" if r["control_index"] >= 0 else "")
                           + f" / {r['stimulus_id']}: clean_margin={r['clean_margin']:.3f} margin={r['margin']:.3f} "
                           f"flip={r['flip']} top1_changed={r['top1_changed']}; clean top-5 "
                           f"{top5(clean[r['stimulus_id']])}; intervened top-5 {top5(r['topk'])}")

    print("[5/5] Checksums, summary, upload ...", flush=True)
    data_files = sorted(p for d in FOLDERS for p in (run.dir / d).glob("*.parquet"))
    (run.dir / "checksums.txt").write_text("\n".join(
        f"{io_mod.sha256_of(p)}  {p.relative_to(run.dir).as_posix()}" for p in data_files) + "\n")
    commits = sorted(rec["git_commit"].unique())
    flagged = [f"{k}: {c['token']!r} ({c['tier_label']})" for k in _run_checks(selection, exp["questions"])
               for c in selection["label_to_present"][k] if c["tier"] >= 3]
    flagged += [f"{c['a_token']!r} <-> {c['b_token']!r} ({c['tier_label']})" for c in selection["big_nonlabel"]
                if c["tier"] >= 3]
    never = selection.get("never_in_top") or {}
    summary_figures = [p.relative_to(run.dir).as_posix() for p in written if "masks" not in p.relative_to(run.dir).parts]
    summary_lines = [
        f"# M3 summary -- run {run.run_id} ({'positives' if positives else 'anomaly'}, position set "
        f"'{exp['position_set']}')", "",
        "## 1. Environment",
        f"- Environment: {pipeline.environment(exp)}",
        f"- git commit(s) that produced the rows: {', '.join(commits)}"
        + (" -- more than one: the run was resumed with --resume after a code change" if len(commits) > 1 else ""),
        f"- resumed: {resumed}; experiment file: {args.experiment}; config_hash (every file in settings/): {run.config_hash()}",
        f"- model: {model_cfg['standin_hf_id'] + ' (stand-in)' if dry else model_cfg['hf_id'] + ' @ ' + model_cfg['revision']}; "
        f"lens: {cfgs['lens_sha']}",
        f"- {m1_line}",
        f"- M2 run: {m2_run} -- treatment pair {pair_name} {pair_words} (pair_score argmax)",
        f"- band: {exp['band']} = layers {figures.layers_text(band_layers)} ({len(band_layers)} layers: {band_layers})",
        f"- position set: '{exp['position_set']}' = classes {list(prompts_mod.POSITION_SETS[exp['position_set']])}; "
        f"skip_first {exp['skip_first']}; alpha {exp['alpha']} (treatment); seed {exp['seed']}; directions {exp['directions']}",
        f"- questions: {exp['questions']}; dataset: {store.name}",
        f"- controls ({controls_source}; rules: src/jlens_spec/controls.py):", *ctl_lines, "",
        "## 2. What passed by assertion / checked by eye / not checked",
        f"- Ran {len(rec)} cells (expected {expected_n} = {len(stim['passages'])} passages x {len(exp['questions'])} "
        f"questions x {len(exp['directions'])} directions x {cells_per_group} cells).",
        f"- Invariant 7 (every treatment cell with all its controls, same run): prompt x direction groups missing a "
        f"cell: {incomplete or 'none'}.",
        "- Edits outside the planned positions: none (any would have stopped the run).",
        f"- Planned (layer, position) entries whose stream did not change: {int(rec['n_planned_unchanged'].sum())} "
        "over all cells (identity cells are never counted -- they edit nothing by design).",
        "- The two directions (reported, not failed): identity cells should be identical; swap, random_direction "
        "and big_nonlabel are the same edit in both directions (the swap is symmetric), so they should match up "
        "to rounding. Only label_to_present differs by direction. Log-probabilities are stored in 16-bit, whose "
        "steps are up to 0.03 at these sizes; on the bf16 27B model, rounding inside the network can add more "
        "(the stored margins and flips are computed in 32-bit before storage).", *direction_lines,
        "- Each control's measured edit size vs the treatment's (same prompt and direction; mean over planned "
        "positions x layers):", *size_lines,
        "- NOT checked: any scientific reading of these numbers.", "",
        "## 3. Figures",
        f"- {len(written)} figure files in {run.dir}/figures/png/: panel_c, margin_vs_deltac_<question>, "
        "flip_heatmap (flip rate per question x kind, flipped/total in each box), margin_change (how far each "
        "edit moved the answer, per question), and one mask figure per prompt (masks/, 5 kinds x 2 directions, "
        f"drawn from where the stream actually changed). The {len(summary_figures)} summary figures are uploaded "
        "with the run; the mask figures are not. Redraw any of them from the data: "
        f"`python scripts/make_figures.py {run.dir} --format pdf`. Combined figures with the other run of this "
        "position set: scripts/combine_panel_c.py.", "",
        "## 4. Anomalies / open questions",
        f"- Controls below the 75% bar or not lowering the label (flagged, used anyway): {flagged or 'none'}",
        "- Treatment tokens never in the lens top-100 at the edited positions within the band (there the coverage "
        "rule sets no bar, and the swap exchanges a token the lens does not read out): "
        + ("; ".join(f"{w!r} in {', '.join(g)}" for w, g in never.items()) if never else "none"),
        f"- Background uploads: {uploader.n_uploads} ok, {uploader.n_failures} failed"
        + (f"; last error: {bg_error}" if bg_error else ""),
        f"- Files downloaded from the store at the end (computed on another machine): {len(downloaded)}", "",
        "## 5. Per-control flip rate and mean margin", "",
        "| kind | # | control | question | direction | flip rate | mean margin | n |", "|---|---|---|---|---|---|---|---|",
        *[f"| {r.kind} | {r.control_index} | {r.control!r} | {r.question_key} | {r.direction} | {r.flip_rate:.3f} | "
          f"{r.mean_margin:.3f} | {r.n} |" for r in per_control.itertuples(index=False)], "",
        *sampled, "",
        "## 7. Checksums",
        f"- {len(data_files)} data files; sha256 of each in checksums.txt (sha256 {io_mod.sha256_of(run.dir / 'checksums.txt')})",
    ]
    url = io_mod.finalize_run(run.dir, summary_lines, store=store, figures_to_upload=summary_figures)
    if url is None:
        print(f"M3 computed everything, but the upload FAILED, so this run counts as unfinished "
              f"(see {run.dir}/upload_error.txt). Run the same command again to retry.")
    else:
        print(f"M3 run complete. See {run.dir}/summary.md")


if __name__ == "__main__":
    main()
