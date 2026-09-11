"""M1 (Phase B, GPU box; after M0 sign-off, scripts/m1_tokens.py, and scripts/download.py). Lens
validation, spec M1 steps 1-4: final-layer agreement, readout reproduction, the causal positive
control (exactly as in the paper), and band signatures. Step 5 (the real-tokenizer single-token
table) runs locally in scripts/m1_tokens.py.

Every run gets its own folder, runs/M1/validate_<UTC start time>/, with a copy of its settings.
Settings come from --experiment (default configs/experiments/m1_validate.yaml). Every check saves
the top-`save_topk` tokens (and check 4 the rank of the model's real next token); the k each check
is judged at is applied afterwards from those files (jlens_spec/m1_checks.py). Nothing under
configs/ is written.

NOT executed yet.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from jlens_spec import env

env.bootstrap()

from jlens_spec import figures  # noqa: E402
from jlens_spec import io as io_mod  # noqa: E402
from jlens_spec import lens as lens_mod  # noqa: E402
from jlens_spec import m1_checks  # noqa: E402
from jlens_spec import model as model_mod  # noqa: E402
from jlens_spec import prompts as prompts_mod  # noqa: E402
from jlens_spec import runs as runs_mod  # noqa: E402

DEFAULT_EXPERIMENT = "configs/experiments/m1_validate.yaml"
SETTINGS_FILES = ["configs/model.yaml", "configs/lens.yaml", "configs/tokens.yaml",
                  "configs/prompt_format.yaml", "stimuli/stimuli.json"]


def _download_dir(download_run) -> Path:
    found = sorted(p.name for p in Path("runs/M1").glob("download_*") if p.is_dir())
    if not download_run:
        raise SystemExit("configs/experiments/m1_validate.yaml: set download_run to the folder name of "
                         f"your scripts/download.py run. Found in runs/M1/: {found or 'none'}")
    d = Path("runs/M1") / download_run
    if not (d / "lens_resolved.yaml").exists():
        raise SystemExit(f"{d}/lens_resolved.yaml not found. Download runs in runs/M1/: {found or 'none'}")
    return d


def _step(n: int, msg: str) -> None:
    print(f"[{n}/5] {msg}", flush=True)


def _top5(row) -> list:
    return [(str(t), round(float(v), 3)) for t, v in zip(list(row["topk_text"])[:5], list(row["topk_logits"])[:5])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help=f"experiment file (default {DEFAULT_EXPERIMENT})")
    args = ap.parse_args()
    env.require_env("HF_TOKEN")
    with open(args.experiment) as f:
        download_dir = _download_dir(yaml.safe_load(f).get("download_run"))  # fail before creating a run

    run = runs_mod.start_run("M1", args.experiment, SETTINGS_FILES)
    print(f"Run folder: {run.dir}", flush=True)
    exp = run.experiment
    model_cfg, lens_cfg = run.load("model.yaml"), run.load("lens.yaml")
    tokens_raw, stim = run.load("tokens.yaml"), run.load("stimuli.json")
    save_k = exp["save_topk"]
    resolved = yaml.safe_load((download_dir / "lens_resolved.yaml").read_text())
    if lens_cfg.get("revision_sha"):
        sha_source = "configs/lens.yaml"
    else:
        lens_cfg = {**lens_cfg, "revision_sha": resolved["revision_sha"]}
        sha_source = f"{download_dir}/lens_resolved.yaml (configs/lens.yaml's revision_sha is empty)"

    _step(1, f"Loading {model_cfg['hf_id']} and the lens ...")
    model = model_mod.load_model(model_cfg, standin=False)
    lens = lens_mod.load_lens(lens_cfg, device=model_mod.device_of(model))
    tok = model.tokenizer
    fmt = prompts_mod.make_fmt(tok, stim, run.load("prompt_format.yaml"))
    all_prompts = [prompts_mod.build_prompt(s, q, fmt) for s in stim["passages"] for q in stim["questions"]]

    def save(rows, name):
        df = pd.DataFrame(rows).assign(run_id=run.run_id)
        df.to_parquet(run.dir / name)
        return df

    _step(2, "Check 1 (final-layer agreement) and check 2 (readout reproduction) ...")
    c1 = exp["check1_final_layer_agreement"]
    df1 = save(m1_checks.check1_final_layer(model, lens, all_prompts[:c1["n_prompts"]], c1["n_positions"],
                                            c1["seed"], save_k), "check1_final_layer.parquet")
    s1 = m1_checks.summarize_check1(df1, c1["overlap_k"])

    c2 = exp["check2_readout"]
    p2 = next(p for p in all_prompts if p.stimulus_id == c2["stimulus"] and p.question_key == c2["question"])
    lang = next(s["matrix_lang"] for s in stim["passages"] if s["id"] == c2["stimulus"])
    lang_ids = {tok.encode(t, add_special_tokens=False)[0] for t in tokens_raw["language_tokens"][lang]}
    df2 = save(m1_checks.check2_readout(model, lens, p2, c2["n_layers"], save_k), "check2_readout.parquet")
    s2 = m1_checks.summarize_check2(df2, lang_ids, c2["hit_k"])

    _step(3, "Check 3 (causal positive control, as in the paper) ...")
    pc = exp["check3_positive_control"]
    r3 = m1_checks.check3_positive_control(model, lens, pc, save_k)
    df3 = save(r3["results"], "check3_results.parquet")
    save(r3["masks"], "check3_masks.parquet")
    save(r3["changes"], "check3_changes.parquet")
    info = r3["info"]
    with open(run.dir / "check3_positive_control.yaml", "w") as f:
        yaml.safe_dump(info, f, sort_keys=False, allow_unicode=True)

    head = [
        f"# M1 summary -- run {run.run_id}" + ("" if info["passed"] else " -- STOPPED"), "",
        "## 1. Environment",
        "- Environment: cuda (Phase B)",
        f"- git commit: {run.manifest()['git_commit']}",
        f"- model: {model_cfg['hf_id']} @ {model_cfg['revision']}",
        f"- lens: {lens_cfg['repo']} / {lens_cfg['filename']} @ {lens_cfg['revision_sha']} (from {sha_source})",
        f"- download run: {download_dir}",
        f"- experiment file: {args.experiment}; config_hash (every file in settings/): {run.config_hash()}",
        f"- saved per check: top-{save_k} tokens", "",
        "## 2. Checks",
        f"- Check 1 (final-layer lens readout vs the model's actual logits, top-{s1['k']} overlap, "
        f"{s1['n_positions']} positions): min={s1['min']}, median={s1['median']}, max={s1['max']}; "
        f"max |logit diff| = {s1['max_abs_logit_diff']:.4g}.",
        f"- Check 2 ({c2['stimulus']}/{c2['question']} matrix positions): fraction with a {lang} language "
        f"token in the readout's top-{s2['k']}, by layer: {s2['fraction_by_layer']}. "
        f"Figure: figures/png/readout_top1_{c2['stimulus']}.png",
        f"- Check 3 (causal positive control, exactly as in the paper): prompt `{info['prompt']}` (raw, "
        f"{info['n_tokens']} tokens: {' | '.join(repr(t) for t in info['tokens'])}); swap {info['pairs']} at EVERY "
        f"token position (deliberate departure from invariant 3 for this control) across layers "
        f"{info['layers'][0]}..{info['layers'][-1]}. passed = {info['passed']}.",
        f"  - clean: top-1 is {pc['expect_clean']}: {info['clean_top1_is_expected']}; top-5 {_top5(df3.iloc[0])}",
        *[f"  - {r['condition']}: top-1 is {pc['expect_swapped']}: {r['top1_is_expected_swapped']}; top-5 {_top5(r)}"
          for _, r in df3.iloc[1:].iterrows()],
        "  - where the stream actually changed was measured at every position (check3_changes.parquet); "
        "nothing changed outside the planned positions. Planned positions that did NOT change: "
        f"{len(info['planned_positions_unchanged'])}" + (f" {info['planned_positions_unchanged'][:10]}"
                                                         if info['planned_positions_unchanged'] else "") + ".",
        f"  - mask figures: figures/png/masks/check3_*.png",
    ]
    credit = (download_dir / "CREDIT.md").read_text() if (download_dir / "CREDIT.md").exists() else "(missing)"
    tail = ["", "## Lens CREDIT.md (from the download run)", credit, "", "## Artifact URL"]

    if not info["passed"]:
        figures.make_figures(run.dir)  # the check 2 readout grid and the check 3 masks still get drawn
        io_mod.finalize_run(run.dir, head + ["", "## STOPPED",
                                             f"Causal positive control failed at alphas {info['alphas_run']}. "
                                             "Do not run M2. Details: check3_positive_control.yaml, "
                                             "check3_results.parquet."] + tail)
        print("STOP: causal positive control failed. Do not run M2.", file=sys.stderr)
        sys.exit(1)

    _step(4, "Check 4 (band signatures: CKA, next-token agreement, kurtosis) ...")
    c4 = exp["check4_band_signatures"]
    cka = save(m1_checks.cka_rows(model, lens, c4["cka_n_tokens"], c4["cka_seed"]), "cka.parquet")
    pos4 = m1_checks.check4_positions(model, lens, all_prompts).assign(run_id=run.run_id)
    pos4.to_parquet(run.dir / "check4_positions.parquet")
    band = m1_checks.band_signatures(pos4, cka, c4["agreement_k"])
    band.to_parquet(run.dir / "band_signatures.parquet")
    onset = m1_checks.motor_onset(band, tuple(c4["motor_onset_mid_range"]))
    (run.dir / "figure_params.json").write_text(json.dumps({"motor_onset": onset}, indent=2))

    _step(5, "Figures (from the saved files), summary, upload ...")
    written = figures.make_figures(run.dir)
    summary_lines = head + [
        f"- Check 4 (band signatures): candidate motor onset = {onset}. The human reads "
        "figures/png/band_signatures.png and fills configs/bands.yaml -- no code does this.", "",
        "## 3. Figures",
        f"- {len(written)} figures in figures/png/, all drawn from this folder's parquet/JSON files. Figures "
        f"are NOT uploaded; redraw them from the data without the model: `python scripts/make_figures.py "
        f"{run.dir} --format pdf`", "",
        "## 4. Anomalies / open questions",
        f"- Check 2 counts a {lang} language token by the FIRST token of each variant in configs/tokens.yaml, "
        "including variants that are several tokens long.", "",
        "## 5. Parquet sha256s",
        *[f"- {n} sha256: {io_mod.sha256_of(run.dir / n)}" for n in
          ("check1_final_layer.parquet", "check2_readout.parquet", "check3_results.parquet",
           "check4_positions.parquet", "band_signatures.parquet", "cka.parquet")],
    ] + tail
    url = io_mod.finalize_run(run.dir, summary_lines)
    if url is None:
        print(f"M1 computed everything, but the upload FAILED, so this run counts as unfinished "
              f"(see {run.dir}/upload_error.txt).")
    else:
        print(f"M1 validation complete. See {run.dir}/summary.md")


if __name__ == "__main__":
    main()
