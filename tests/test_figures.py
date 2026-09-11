"""figures.make_figures builds every milestone's figures from a run folder's saved files alone (no
model), in any format, each format in its own figures/<format>/ folder. Inputs are synthetic but
written with the same writers the milestone scripts use, so the parquet round-trip (list columns ->
numpy arrays, None in bool columns) is exercised."""
import json

import pandas as pd
import pytest

from jlens_spec import figures
from jlens_spec import io as io_mod


def _files(paths):
    return sorted(p.name for p in paths)


def test_m0_masks_from_parquet(tmp_path):
    rows = []
    for sid in ("sp_01", "fr_01"):
        for i, (text, cls) in enumerate([("<|im_start|>", "template"), ("What", "question"),
                                         (" language", "question"), ("\n\n", "template"),
                                         ("El", "matrix"), (" oiseau", "intrusion"), ("", "template")]):
            rows.append({"stimulus_id": sid, "question_key": "report", "pos": i, "token_id": 100 + i,
                         "token_text": text, "class": cls, "edited": cls == "question",
                         "is_metric_pos": i == 6})
    io_mod.append_records(rows, tmp_path / "masks.parquet")

    written = figures.make_figures(tmp_path, formats=("png", "pdf", "svg"), milestone="M0")
    assert _files(written) == sorted(f"{sid}_report.{fmt}" for sid in ("sp_01", "fr_01")
                                     for fmt in ("png", "pdf", "svg"))
    # each format in its own folder, never side by side
    assert all(p.parent == tmp_path / "figures" / p.suffix[1:] / "masks" and p.stat().st_size > 0
               for p in written)


def test_m1_from_parquet(tmp_path):
    layers = [3, 7, 11]
    pd.DataFrame([{"stimulus_id": "sp_01", "question_key": "report", "pos": pos, "layer": l,
                   "topk_ids": [1, 2, 3], "topk_text": [" Spanish", "a", ""], "topk_logits": [3.0, 2.0, 1.0]}
                  for pos in (10, 11) for l in layers]).to_parquet(tmp_path / "check2_readout.parquet")
    pd.DataFrame([{"stimulus_id": "antonym", "question_key": "swap_alpha1", "pos": i, "token_id": 100 + i,
                   "token_text": t, "class": "prompt", "edited": True, "is_metric_pos": i == 3}
                  for i, t in enumerate(['"', '小', '"的', '反义词是"'])]).to_parquet(tmp_path / "check3_masks.parquet")
    pd.DataFrame([{"layer_a": a, "layer_b": b, "cka": 1.0 if a == b else 0.5}
                  for a in layers for b in layers]).to_parquet(tmp_path / "cka.parquet")
    pd.DataFrame({"layer": layers, "cka_onset_score": [0.1, 0.2, float("nan")],
                  "agreement_top1": [0.0, 0.2, 0.9], "agreement_top5": [0.1, 0.4, 1.0],
                  "kurtosis": [1.0, 2.0, 3.0]}).to_parquet(tmp_path / "band_signatures.parquet")
    (tmp_path / "figure_params.json").write_text(json.dumps({"motor_onset": 11}))

    written = figures.make_figures(tmp_path, formats=("pdf",), milestone="M1")
    assert _files(written) == ["band_signatures.pdf", "check3_swap_alpha1.pdf", "cka_heatmap.pdf",
                               "readout_top1_sp01.pdf"]
    assert tmp_path / "figures" / "pdf" / "masks" / "check3_swap_alpha1.pdf" in written


def test_m2_from_parquet(tmp_path):
    rows = [{"stimulus_id": sid, "question_key": q, "pos": pos, "class": cls, "layer": l,
             "token": tok, "cos": 0.1 * l, "rank": 5}
            for sid in ("sp_01", "fr_01") for q in ("report", "anomaly")
            for pos, cls in ((4, "question"), (9, "matrix")) for l in (3, 4)
            for tok in ("Spanish", "French")]
    io_mod.append_records(rows, tmp_path / "loadings.parquet")
    (tmp_path / "figure_params.json").write_text(json.dumps({
        "band": [3, 4],
        "loading_heatmaps": [{"stimulus_id": "sp_01", "token": "Spanish"},
                             {"stimulus_id": "fr_01", "token": "French"}]}))

    written = figures.make_figures(tmp_path, milestone="M2")
    assert _files(written) == ["loading_heatmap_fr_01.png", "loading_heatmap_sp_01.png",
                               "loading_summary_bars.png"]


def test_m3_from_parquet_round_trip(tmp_path):
    """The bug this guards: rows read back from parquet carry intervention_logs as numpy arrays
    (`if not logs` raised) and flip as None/bool objects."""
    rows = []
    for sid in ("sp_01", "fr_01"):
        for q in ("anomaly", "report"):
            for kind in ("identity", "swap", "random_direction"):
                logs = [] if kind == "identity" else [
                    {"pos": 5, "layer": 3, "c_before": [0.1, 0.2], "c_after": [0.2, 0.1],
                     "delta_c_norm": 0.3, "delta_h_norm": 1.0, "alpha": 1.0, "kind": kind}]
                rows.append({"stimulus_id": sid, "question_key": q, "direction": "m2i", "kind": kind,
                             "flip": None if kind == "random_direction" and sid == "fr_01" else kind == "swap",
                             "margin": 0.5, "intervention_logs": logs, "logprobs_fp16": [0.0] * 8})
    io_mod.append_records(rows, tmp_path / "records.parquet")

    written = figures.make_figures(tmp_path, formats=("png", "svg"), milestone="M3")
    assert _files(written) == sorted(f"{n}.{fmt}" for n in ("panel_c", "margin_vs_deltac_anomaly",
                                                             "margin_vs_deltac_report")
                                     for fmt in ("png", "svg"))


def test_missing_inputs_are_skipped_not_fatal(tmp_path, capsys):
    assert figures.make_figures(tmp_path, milestone="M3") == []
    assert "records.parquet not found" in capsys.readouterr().err


def test_unknown_format_raises(tmp_path):
    io_mod.append_records([{"stimulus_id": "sp_01", "question_key": "report", "pos": 0, "token_id": 1,
                            "token_text": "a", "class": "template", "edited": False,
                            "is_metric_pos": True}], tmp_path / "masks.parquet")
    with pytest.raises(ValueError, match="unknown figure format"):
        figures.make_figures(tmp_path, formats=("jpg",), milestone="M0")


def test_out_dir_writes_straight_into_that_folder(tmp_path):
    pd.DataFrame([{"layer_a": a, "layer_b": b, "cka": 1.0} for a in (1, 2) for b in (1, 2)]
                 ).to_parquet(tmp_path / "cka.parquet")
    out = tmp_path / "writeup"
    written = figures.make_figures(tmp_path, formats=("pdf", "png"), milestone="M1", out_dir=out)
    assert sorted(written) == [out / "cka_heatmap.pdf", out / "cka_heatmap.png"]
    assert not (tmp_path / "figures").exists()


def test_milestone_comes_from_the_run_manifest(tmp_path):
    io_mod.write_manifest(tmp_path, milestone="M3", run_id="x")
    assert figures.make_figures(tmp_path) == []  # dispatched to M3 (records.parquet missing -> skipped)


def test_folder_without_manifest_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="no manifest.json"):
        figures.make_figures(tmp_path)
