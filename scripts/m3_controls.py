"""M3 control tokens (Phase B): one CONTROLS run per position set, before that position set's positives
run. Picks the control tokens by rule (src/jlens_spec/controls.py -- module docstring) from the M2
run's saved top-100 readout plus one clean pass over all 64 prompts, saves them in a new run folder,
uploads it, and STOPS: no cell runs.

You then read its summary.md: every pick, the next candidates of each check, and every token the
exclusion rules removed. If a pick carries language (a Spanish or French word, a place name, ...),
add it to controls.control_ineligible in configs/tokens.yaml, commit, and run this again -- a new run
folder with new picks. When you accept a run, put its folder name into `controls_run:` of the
position set's positives experiment file (configs/experiments/m3_positives_<position set>.yaml).

    python scripts/m3_controls.py --experiment configs/experiments/m3_controls_question.yaml
    python scripts/m3_controls.py --experiment configs/experiments/m3_controls_message.yaml

Saved in runs/M3/controls_<position set>_<time>/controls/:
    selection.yaml      the picks (3 label_to_present tokens per check, 3 big_nonlabel pairs), the
                        next candidates of each check, and the settings a positives run must match
    candidates.parquet  every candidate with its per-check coverage, |Δc| and "lowers the label", plus
                        every excluded token with its reason
    pairs.parquet       every big_nonlabel pair of the pool with its per-check |Δc| after scaling
Every launch starts a new run (it takes minutes; after a crash, just run it again)."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml

from jlens_spec import env

env.bootstrap()

from jlens_spec import controls as controls_mod  # noqa: E402
from jlens_spec import figures  # noqa: E402
from jlens_spec import io as io_mod  # noqa: E402
from jlens_spec import metrics  # noqa: E402
from jlens_spec import pipeline  # noqa: E402
from jlens_spec import prompts as prompts_mod  # noqa: E402
from jlens_spec import runs as runs_mod  # noqa: E402

N_RUNNERS_UP = 3        # per check: the label_to_present candidates listed after the picks
NEVER_CONTROLS = ("special token", "does not re-tokenize to itself")
CONTROL_FILES = ("controls/selection.yaml", "controls/candidates.parquet", "controls/pairs.parquet")


def _py(v):
    """numpy scalars/arrays -> plain Python, for yaml."""
    if isinstance(v, dict):
        return {str(k): _py(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_py(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, float) and v != v:
        return None
    return v


def _float32(df: pd.DataFrame) -> pd.DataFrame:
    """The control tables are stored in 32-bit floats (half the size; 7 significant digits is plenty)."""
    return df.astype({c: "float32" for c in df.select_dtypes("float64").columns})


def _entry(i: int, row: dict, checks: list[str], fields: tuple) -> dict:
    """One picked control for selection.yaml: its summary fields, then its value of each `fields`
    entry in every check nested under per_check (so the file stays readable with 16 checks)."""
    per = {f"{f}_{k}" for f in fields for k in checks}
    out = {"control_index": i, **{k: v for k, v in row.items() if k not in per}}
    out["per_check"] = {k: {f: row[f"{f}_{k}"] for f in fields} for k in checks}
    return out


def _select_controls(run, exp, model, lens, stim, fmt, tokens_raw, pair_words, band_layers, m2_dir, store):
    """Pick the controls (controls.py) and save them in controls/. Returns (the selection dict, the
    classification of every token in the pool: token_id, token, excluded_reason)."""
    c = exp["controls"]
    classes = prompts_mod.POSITION_SETS[exp["position_set"]]
    tok = model.tokenizer
    questions = list(stim["questions"])
    stimuli = {s["id"]: s for s in stim["passages"]}
    keys = [(s["id"], q) for s in stim["passages"] for q in questions]
    for sid, q in keys:
        pipeline.fetch(m2_dir / "topk" / f"{sid}_{q}.parquet", store)
    topk = pq.read_table(m2_dir / "topk", columns=["stimulus_id", "question_key", "pos", "class", "layer", "topk_ids"],
                         filters=[("layer", "in", band_layers), ("class", "in", list(classes))])
    cov = controls_mod.coverage(topk, classes, band_layers, exp["skip_first"], c["pool_k"])
    groups = [controls_mod.group_name(stimuli[sid]["matrix_lang"], q) for sid, q in keys]
    by_group = {}
    for g, key in zip(groups, keys):
        by_group.setdefault(g, []).append(key)
    cov_group = controls_mod.coverage_by_group(cov, by_group, len(band_layers))

    prompts, masks = [], []
    for sid, q in keys:
        p = prompts_mod.build_prompt(stimuli[sid], q, fmt)
        prompts.append(p)
        masks.append(prompts_mod.mask(p, set(classes), skip_first=exp["skip_first"]))
    answers = tokens_raw["answers"]
    answer_ids = metrics.answer_ids(tok, answers["yes"] + answers["no"]
                                    + [w for forms in answers["hello"].values() for w in forms])
    info = controls_mod.classify_tokens(sorted(cov["token_id"].unique()), tok,
                                        tokens_raw["controls"]["control_ineligible"], answer_ids,
                                        controls_mod.passage_words(prompts, tok))
    eligible = info[info["excluded_reason"] == ""].reset_index(drop=True)
    pair_ids = [tok.encode(w, add_special_tokens=False)[0] for w in pair_words]
    cands, treat = controls_mod.with_coverage(eligible, cov_group, pair_words, pair_ids, questions)
    pool = controls_mod.big_nonlabel_pool(cands, treat, c["pair_pool"], c["bars"])

    print(f"      clean pass over {len(prompts)} prompts: |Δc| and direction for {len(cands)} label_to_present "
          f"candidates (picked separately for each of the {len(treat)} row x question checks, lowering the label), "
          f"lens distances for {len(pool) * (len(pool) - 1) // 2} big_nonlabel pairs (one set, every check) ...",
          flush=True)
    est = controls_mod.estimate_delta_c(model, lens, prompts, masks, band_layers, cands["token_id"].tolist(),
                                        pair_ids, pool["token_id"].tolist())
    cands, treat, pair_tab = controls_mod.with_delta_c(cands, treat, est, groups, pair_words,
                                                       pool["token_id"].tolist(), c["norm_scale"])
    n = c["n_per_kind"]
    ranked = controls_mod.select_label_to_present(cands, treat, n + N_RUNNERS_UP, c["bars"])
    ltp = {k: df.head(n) for k, df in ranked.items()}
    runners_up = {k: df.iloc[n:] for k, df in ranked.items()}
    bn = controls_mod.select_big_nonlabel(pool, pair_tab, treat, n, c["bars"])
    if any(df.empty for df in ltp.values()) or bn.empty:
        raise SystemExit("a label_to_present or big_nonlabel control could not be formed at all (no eligible "
                         "candidate) -- a treatment without its controls breaks invariant 7")

    (run.dir / "controls").mkdir(exist_ok=True)
    excluded = info[info["excluded_reason"] != ""]
    io_mod.write_parquet(_float32(pd.concat([cands, excluded], ignore_index=True)),
                         run.dir / "controls" / "candidates.parquet")
    io_mod.write_parquet(_float32(pair_tab), run.dir / "controls" / "pairs.parquet")
    checks = list(treat)

    def listed(frames):  # per check: its tokens, control_index counting from 0
        return {k: [{"control_index": i, **r} for i, r in enumerate(df.to_dict("records"))] for k, df in frames.items()}

    selection = {
        "selection_version": controls_mod.SELECTION_VERSION,
        "picked_by_run": run.run_id, "git_commit": run.manifest()["git_commit"],
        "position_set": exp["position_set"], "position_classes": list(classes),
        "band": exp["band"], "band_layers": band_layers, "skip_first": exp["skip_first"],
        "from_m2_run": m2_dir.name, "pair_words": list(pair_words), "pool_k": c["pool_k"],
        "bars": c["bars"], "pair_pool": c["pair_pool"], "norm_scale": c["norm_scale"],
        "checks": treat,
        # per check: the n tokens that check's label_to_present cells use
        "label_to_present": listed(ltp),
        # per check: the next candidates after the picks -- listed for the review, never used by a cell
        "label_to_present_runners_up": listed(runners_up),
        "big_nonlabel": [_entry(i, r, checks, ("dc", "dc_ratio")) for i, r in enumerate(bn.to_dict("records"))],
        "never_in_top": controls_mod.never_in_top(treat),
        "n_candidates_eligible": int(len(cands)),
        "excluded_counts": excluded["excluded_reason"].value_counts().to_dict(),
        "excluded_special": excluded[excluded["excluded_reason"].isin(NEVER_CONTROLS)][
            ["token_id", "token", "excluded_reason"]].to_dict("records"),
    }
    with open(run.dir / "controls" / "selection.yaml", "w") as f:
        yaml.safe_dump(_py(selection), f, sort_keys=False, allow_unicode=True)
    return yaml.safe_load((run.dir / "controls" / "selection.yaml").read_text()), info


def _runner_up_lines(selection) -> list[str]:
    return [f"  - {k}: " + ("; ".join(f"{c['token']!r} ({c['tier_label']}, |Δc| {c['dc_ratio']:.2f}x, "
                                      f"{controls_mod.coverage_text(c['cov_ratio'])})" for c in runners)
                             or "none left")
            for k, runners in selection["label_to_present_runners_up"].items()]


def _excluded_lines(info: pd.DataFrame) -> list[str]:
    """Every token an exclusion rule removed from the pool, by rule; the tokens that can never be
    controls (special / not re-tokenizing) only counted (listed in selection.yaml excluded_special)."""
    ex = info[info["excluded_reason"] != ""]
    lines = []
    for reason, g in ex.groupby("excluded_reason", sort=True):
        if reason in NEVER_CONTROLS:
            lines.append(f"- {reason}: {len(g)} (listed in controls/selection.yaml, excluded_special)")
        else:
            lines.append(f"- {reason} ({len(g)}): " + ", ".join(repr(t) for t in sorted(g["token"])))
    return lines or ["- none"]


def _check_experiment(exp, path, store) -> tuple[str, str]:
    """Refuse bad settings before any work: a controls experiment file, a known position set, the M1
    run passed (real runs), a finished M2 run. Returns (the M1 summary line, the M2 run name)."""
    if "controls" not in exp or "questions" in exp:
        raise SystemExit(f"{path} is not a controls experiment file (it needs a `controls:` block and no "
                         "`questions:`) -- e.g. configs/experiments/m3_controls_question.yaml")
    if exp["position_set"] not in prompts_mod.POSITION_SETS:
        raise SystemExit(f"position_set must be one of {sorted(prompts_mod.POSITION_SETS)}")
    m1_line = pipeline.check_m1_passed(exp, store)
    if not exp.get("from_m2_run"):
        raise SystemExit("set from_m2_run in the experiment file (the finished M2 run, runs/M2/<this>)")
    m2_run = pipeline.resolve_run(exp["from_m2_run"], "M2", exp, store)
    pipeline.require_finished(pipeline.runs_root(exp) / "M2" / m2_run, store, "M2 run")
    return m1_line, m2_run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True,
                    help="a controls experiment file, e.g. configs/experiments/m3_controls_question.yaml")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu",
                    help="dry runs only: mps or cpu (a real run uses the GPU via device_map='auto')")
    args = ap.parse_args()

    exp0 = pipeline.read_experiment(args.experiment)
    dry, store, root = pipeline.is_dryrun(exp0), pipeline.store_for(exp0), pipeline.runs_root(exp0)
    if not dry:
        env.require_env("HF_TOKEN")
    # Checks that need no model, before any run folder is created (repeated on the run's settings copy).
    _check_experiment(exp0, args.experiment, store)
    with open(pipeline.settings_files(exp0)[-1]) as f:
        env.check_bands(yaml.safe_load(f))

    run = runs_mod.start_run("M3", args.experiment, pipeline.settings_files(exp0), root=root)
    print(f"New run folder: {run.dir}", flush=True)
    exp = run.experiment
    m1_line, m2_run = _check_experiment(exp, args.experiment, store)
    m2_dir = root / "M2" / m2_run
    model_cfg, lens_cfg, tokens_raw = run.load("model.yaml"), run.load("lens.yaml"), run.load("tokens.yaml")
    stim, prompt_format, bands = run.load("stimuli.json"), run.load("prompt_format.yaml"), run.load("bands.yaml")
    m2 = yaml.safe_load(pipeline.fetch(m2_dir / "pair_scores.yaml", store).read_text())
    pair_name, pair_words = m2["pair_score_argmax"], m2["pair_words"]
    if tokens_raw["pairs"].get(pair_name) != pair_words:
        raise SystemExit(f"M2's winning pair {pair_name} = {pair_words} is not in this run's tokens.yaml pairs")

    print(f"[1/3] Loading the model and the lens ({pipeline.environment(exp)}) ...", flush=True)
    model, lens = pipeline.load_model_and_lens(exp, model_cfg, lens_cfg, args.device)
    band_layers = env.check_bands(bands, lens.layers)[exp["band"]]
    fmt = prompts_mod.make_fmt(model.tokenizer, stim, prompt_format)

    print("[2/3] Control tokens (all 16 row x question checks) ...", flush=True)
    selection, info = _select_controls(run, exp, model, lens, stim, fmt, tokens_raw, pair_words, band_layers,
                                       m2_dir, store)
    ctl_lines = controls_mod.control_lines(selection, list(stim["questions"]))
    print("\n".join(ctl_lines), flush=True)

    print("[3/3] Summary, upload ...", flush=True)
    flagged = [f"{k}: {c['token']!r} ({c['tier_label']})" for k, picks in selection["label_to_present"].items()
               for c in picks if c["tier"] >= 3]
    flagged += [f"{c['a_token']!r} <-> {c['b_token']!r} ({c['tier_label']})" for c in selection["big_nonlabel"]
                if c["tier"] >= 3]
    never = selection.get("never_in_top") or {}
    positives_file = f"configs/experiments/{'dryrun/' if dry else ''}m3_positives_{exp['position_set']}.yaml"
    summary_lines = [
        f"# M3 controls -- run {run.run_id} (position set '{exp['position_set']}')", "",
        "No cell has run. Read the picks below; accept them or pick again (section 5).", "",
        "## 1. Environment",
        f"- Environment: {pipeline.environment(exp)}",
        f"- git commit: {run.manifest()['git_commit']}",
        f"- experiment file: {args.experiment}; config_hash (every file in settings/): {run.config_hash()}",
        f"- model: {model_cfg['standin_hf_id'] + ' (stand-in)' if dry else model_cfg['hf_id'] + ' @ ' + model_cfg['revision']}; "
        f"lens: {'random-lens-seed-' + str(exp['dryrun']['lens_seed']) if dry else lens_cfg['revision_sha']}",
        f"- {m1_line}",
        f"- M2 run: {m2_run} -- treatment pair {pair_name} {pair_words} (pair_score argmax)",
        f"- band: {exp['band']} = layers {figures.layers_text(band_layers)} ({len(band_layers)} layers: {band_layers})",
        f"- position set: '{exp['position_set']}' = classes {list(prompts_mod.POSITION_SETS[exp['position_set']])}; "
        f"skip_first {exp['skip_first']}", "",
        "## 2. The picks (rules: src/jlens_spec/controls.py)",
        "- label_to_present: 3 tokens for each of the 16 checks (row x question; a cell uses its own check's); "
        "big_nonlabel: 3 pairs shared by every check. Both runs of this position set use these.",
        *ctl_lines,
        f"- Below the 75% bar or not lowering the label (flagged, would be used anyway): {flagged or 'none'}",
        "- Treatment tokens never in the lens top-100 at the edited positions within the band (there the coverage "
        "rule sets no bar): " + ("; ".join(f"{w!r} in {', '.join(g)}" for w, g in never.items()) if never else "none"),
        "",
        f"## 3. The next {N_RUNNERS_UP} label_to_present candidates of each check",
        "- What would probably replace a pick you exclude (probably: an exclusion can also change the "
        "big_nonlabel pool). Never used by a cell.",
        *_runner_up_lines(selection), "",
        f"## 4. What the exclusion rules removed ({len(info)} tokens were in the top-100 at the edited positions "
        f"within the band; {selection['n_candidates_eligible']} remained)",
        *_excluded_lines(info), "",
        "## 5. What next",
        "- If a pick (or a big_nonlabel member) carries language -- a Spanish or French word, a place name -- add "
        "it to controls.control_ineligible in configs/tokens.yaml, commit, and run the same command again: a new "
        "run folder, new picks.",
        f"- If you accept these: put `{run.run_id}` into `controls_run:` of {positives_file}.", "",
        "## 6. Checksums",
        *[f"- {f} sha256: {io_mod.sha256_of(run.dir / f)}" for f in CONTROL_FILES],
    ]
    url = io_mod.finalize_run(run.dir, summary_lines, store=store)
    if url is None:
        print(f"The controls were picked, but the upload FAILED, so this run counts as unfinished "
              f"(see {run.dir}/upload_error.txt). Run the same command again (a new run).")
    else:
        print(f"Controls picked -- no cell has run. Read {run.dir}/summary.md; to accept them, put {run.run_id} "
              f"into controls_run: of {positives_file}.")


if __name__ == "__main__":
    main()
