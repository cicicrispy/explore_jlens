"""M3 (Phase B): the paper's protocol -- Stage 1 grid. Position set = {"question"} only. Layers =
bands.workspace. alpha=1 for treatment. Refuses to run if bands.workspace, controls.label_to_present
.target, or controls.big_nonlabel.pair is null.

NOT executed yet.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import yaml

from jlens_spec import env

env.bootstrap()

from jlens_spec import figures
from jlens_spec import io as io_mod
from jlens_spec import lens as lens_mod
from jlens_spec import metrics
from jlens_spec import model as model_mod
from jlens_spec import runner

RUN_DIR = Path("runs/M3")
SEED = 0


def _load_yaml(name: str):
    with open(Path("configs") / name) as f:
        return yaml.safe_load(f)


def main() -> None:
    env.require_env("HF_TOKEN")
    model_cfg = _load_yaml("model.yaml")
    lens_cfg = _load_yaml("lens.yaml")
    tokens_raw = _load_yaml("tokens.yaml")
    bands = _load_yaml("bands.yaml")

    workspace = bands.get("workspace")
    assert workspace is not None, "configs/bands.yaml workspace is null -- M3 refuses to run"
    layers = env.expand_band(workspace)

    controls = tokens_raw["controls"]
    assert controls["label_to_present"]["target"] is not None, "controls.label_to_present.target is null -- M3 refuses to run"
    assert controls["big_nonlabel"]["pair"] is not None, "controls.big_nonlabel.pair is null -- M3 refuses to run"

    model = model_mod.load_model(model_cfg, standin=False)
    assert lens_cfg.get("revision_sha"), (
        "configs/lens.yaml revision_sha is null -- copy it from runs/M1/lens_resolved.yaml (M1 step 0)"
    )
    lens = lens_mod.load_lens(lens_cfg, device=env.get_device())
    tokenizer = model.tokenizer
    tokens_cfg = metrics.build_tokens_cfg(tokenizer, tokens_raw)

    with open("stimuli/stimuli.json") as f:
        stim = json.load(f)
    stimuli_by_id = {s["id"]: s for s in stim["passages"]}
    fmt = {"tokenizer": tokenizer, "questions": stim["questions"]}

    # pair {the argmax of pair_score from M2}: M2's controls_proposed.yaml is the source of truth
    # (this script does not recompute pair_score itself).
    with open("runs/M2/controls_proposed.yaml") as f:
        m2_controls = yaml.safe_load(f)
    pair_name = m2_controls["pair_score_argmax"]
    assert pair_name is not None, "M2's pair_score_argmax is null -- cannot select a pair for M3"

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    records_path = RUN_DIR / "records.parquet"

    cells = [
        runner.Cell(
            stimulus_id=stimulus_id,
            question_key=qkey,
            direction=direction,
            pair_name=pair_name,
            kind=kind,
            layers=layers,
            position_set={"question"},
            alpha=1.0,
            seed=SEED,
        )
        for stimulus_id in stimuli_by_id
        for qkey in stim["questions"]
        for direction in ("m2i", "i2m")
        for kind in ("identity", "swap", "label_to_present", "big_nonlabel", "random_direction")
    ]

    cfgs = {
        "stimuli": stimuli_by_id,
        "fmt": fmt,
        "tokens_raw": tokens_raw,
        "tokens_cfg": tokens_cfg,
        "skip_first": 4,
        "config_hash": env.config_hash(
            "configs/model.yaml",
            "configs/lens.yaml",
            "configs/tokens.yaml",
            "configs/bands.yaml",
        ),
        "lens_sha": lens_cfg.get("revision_sha", ""),
        "model_revision": model_cfg.get("revision", ""),
    }

    # Background uploader: syncs runs/M3/records.parquet to the HF dataset as cells complete,
    # without blocking the GPU trace loop in run_grid (see io.BackgroundUploader / the README's
    # "Background uploads" section). flush()+shutdown() below wait for the last of these to land
    # before the final, synchronous upload_run call that also picks up figures/summary.md.
    uploader = io_mod.BackgroundUploader(RUN_DIR)
    all_records = list(runner.run_grid(model, lens, cells, cfgs, records_path, uploader=uploader))
    _, bg_error = uploader.flush()
    uploader.shutdown()
    bg_upload_note = (f"- Background uploads: {uploader.n_uploads} ok, {uploader.n_failures} failed"
                      + (f"; last error: {bg_error}" if bg_error else ""))

    # Built from records.parquet as read back from disk -- same call as scripts/make_figures.py.
    figures.make_figures(RUN_DIR, milestone="M3")

    rng = random.Random(SEED)
    grouped = {}
    for r in all_records:
        grouped.setdefault((r.question_key, r.direction, r.kind), []).append(r)
    sampled_lines = ["## Ten randomly sampled raw cells per {question x direction x kind}", ""]
    for key, recs in grouped.items():
        for r in rng.sample(recs, min(10, len(recs))):
            sampled_lines.append(
                f"- {key}: stimulus={r.stimulus_id} clean_margin={r.clean_margin:.3f} "
                f"margin={r.margin:.3f} flip={r.flip} top1_changed={r.top1_changed} top5={r.top5}"
            )

    expected_n = len(stimuli_by_id) * len(stim["questions"]) * 2 * 1 * 5
    summary_lines = [
        "# M3 summary",
        "",
        "## 1. Environment",
        "- Environment: cuda (Phase B)",
        f"- git commit: {env.git_commit()}",
        f"- workspace band: {workspace}, pair: {pair_name}, seed: {SEED}",
        "",
        "## 2. What passed by assertion / checked by eye / not checked",
        f"- Ran {len(all_records)} cells (expected {expected_n} = "
        f"{len(stimuli_by_id)} stimuli x {len(stim['questions'])} questions x 2 directions x 1 pair x 5 kinds).",
        "- NOT scientifically interpreted here, per instructions -- see panel_c.png and records.parquet.",
        "",
        "## 3. Figures",
        "- runs/M3/figures/panel_c.png",
        "- runs/M3/figures/margin_vs_deltac_anomaly.png",
        "- runs/M3/figures/margin_vs_deltac_report.png",
        "- Built from runs/M3/records.parquet. Regenerate without the model: "
        "`python scripts/make_figures.py --milestone M3 --format pdf`.",
        "",
        "## 4. Anomalies / open questions",
        "- This script has not been executed; the cell count above is the design target, not a "
        "confirmed count from a real run.",
        "",
    ]
    summary_lines += sampled_lines
    summary_lines += [
        "",
        "## 5. Artifact URL, parquet sha256s",
        bg_upload_note,
        f"- runs/M3/records.parquet sha256: {io_mod.sha256_of(records_path)}",
    ]

    # Interim until M3 moves to run folders (stage 3): manifest written here, then the upload.
    io_mod.write_manifest(RUN_DIR, milestone="M3", environment="cuda", n_cells=len(all_records))
    io_mod.finalize_run(RUN_DIR, summary_lines)
    print("M3 grid complete. See runs/M3/summary.md")


if __name__ == "__main__":
    main()
