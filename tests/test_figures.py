"""figures.make_figures builds every milestone's figures from a run folder's saved files alone (no
model), in any format, each format in its own figures/<format>/ folder. Inputs are synthetic but
written with the same writers the milestone scripts use, so the parquet round-trip (list columns ->
numpy arrays, None in bool columns) is exercised."""
import json

import numpy as np
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
                               "readout_top1_sp_01.pdf"]
    assert tmp_path / "figures" / "pdf" / "masks" / "check3_swap_alpha1.pdf" in written


_PAIRS = {"plain": ["Spanish", "French"], "space": [" Spanish", " French"]}


def test_m2_from_parquet(tmp_path):
    """Per prompt a cos grid and a rank grid (every pair's es/fr member x every saved layer) + the bars."""
    for sid in ("sp_01", "fr_01"):
        rows = [{"stimulus_id": sid, "question_key": "report", "pos": pos, "class": cls, "layer": l,
                 "token": tok, "token_id": 1, "cos": 0.1 * l - 0.2, "rank": 5 + 60 * pos}
                for pos, cls in ((0, "template"), (1, "instruction"), (4, "question"), (9, "matrix"))
                for l in (2, 3, 4, 5) for tok in ("Spanish", "French", " Spanish", " French")]
        io_mod.write_parquet(rows, tmp_path / "loadings" / f"{sid}_report.parquet")
    (tmp_path / "figure_params.json").write_text(json.dumps({"band": [3, 4], "pairs": _PAIRS}))

    written = figures.make_figures(tmp_path, milestone="M2")
    assert _files(written) == ["fr_01_report.png", "fr_01_report.png", "loading_summary_bars.png",
                               "sp_01_report.png", "sp_01_report.png"]
    assert tmp_path / "figures" / "png" / "loadings" / "sp_01_report.png" in written
    assert tmp_path / "figures" / "png" / "ranks" / "sp_01_report.png" in written


def test_layers_text_names_every_block_of_a_split_band():
    assert figures.layers_text(range(9, 19)) == "9–18"
    assert figures.layers_text([9, 10, 11, 12, 13, 16, 17, 18]) == "9–13 and 16–18"
    assert figures.layers_text([2, 5, 6, 9]) == "2, 5–6 and 9"


def _m3_run(run_dir, pair="space", lens_sha="L", position_set=("question",)):
    """A synthetic M3 run folder written with the same writers the script uses."""
    for sid in ("sp_01", "fr_01"):
        for q in ("anomaly", "report"):
            name = f"{sid}_{q}"
            recs, details = [], []
            for d in ("m2i", "i2m"):
                for kind, ci in (("identity", -1), ("swap", -1), ("random_direction", -1),
                                 ("label_to_present", 0), ("big_nonlabel", 0)):
                    logs = [] if kind == "identity" else [
                        {"pos": 2, "layer": 3, "c_before": [0.1, 0.2], "c_after": [0.2, 0.1],
                         "delta_c_norm": 0.3, "delta_h_norm": 1.0, "alpha": 1.0, "kind": kind}]
                    recs.append({"stimulus_id": sid, "question_key": q, "direction": d, "kind": kind,
                                 "control_index": ci, "flip": None if kind == "random_direction" and sid == "fr_01"
                                 else kind == "swap", "margin": 0.5, "clean_margin": 0.2, "intervention_logs": logs,
                                 "pair_name": pair, "lens_sha": lens_sha, "model_revision": "main",
                                 "git_commit": "abc", "position_set": list(position_set),
                                 "logprobs_fp16": np.zeros(8, dtype=np.float16)})
                    for pos in range(4):
                        details.append({"kind": kind, "direction": d, "control_index": ci, "layer": 3, "pos": pos,
                                        "planned": pos == 2, "change": 1.0 if pos == 2 and kind != "identity" else 0.0})
            io_mod.write_parquet(io_mod.records_table(recs), run_dir / "records" / f"{name}.parquet")
            io_mod.write_parquet(details, run_dir / "details" / f"{name}.parquet")
            io_mod.write_parquet([{"stimulus_id": sid, "question_key": q, "pos": i, "token_id": 10 + i,
                                   "token_text": t, "class": c, "edited": i == 2, "is_metric_pos": i == 3}
                                  for i, (t, c) in enumerate([("<|im_start|>", "template"), ("I", "instruction"),
                                                              ("What", "question"), ("", "template")])],
                                 run_dir / "tokens" / f"{name}.parquet")
    (run_dir / "figure_params.json").write_text(json.dumps({"band_layers": [3], "position_set": position_set[0]}))
    return run_dir


def test_m3_from_parquet_round_trip(tmp_path):
    """The bug this guards: rows read back from parquet carry intervention_logs as numpy arrays
    (`if not logs` raised) and flip as None/bool objects. Plus one 5x2 mask figure per prompt."""
    _m3_run(tmp_path)
    written = figures.make_figures(tmp_path, formats=("png", "svg"), milestone="M3")
    names = ["panel_c", "flip_heatmap", "margin_change", "margin_vs_deltac_anomaly", "margin_vs_deltac_report"] + \
        [f"{s}_{q}" for s in ("sp_01", "fr_01") for q in ("anomaly", "report")]
    assert _files(written) == sorted(f"{n}.{fmt}" for n in names for fmt in ("png", "svg"))
    assert tmp_path / "figures" / "png" / "masks" / "sp_01_report.png" in written


def test_by_language_draws_each_language_on_the_whole_runs_axes(tmp_path):
    """The summary figures per passage language, from the same records; the scales must be the whole
    run's, or the two languages cannot be read against each other."""
    _m3_run(tmp_path)
    written = figures.m3_by_language(tmp_path, formats=("png",))
    names = [f"{n}_{lang}" for lang in ("es", "fr")
             for n in ("flip_heatmap", "margin_change", "margin_vs_deltac_anomaly", "margin_vs_deltac_report")]
    assert _files(written) == sorted(f"{n}.png" for n in names)

    df = figures.read_records(tmp_path)
    df.loc[df["stimulus_id"].str.startswith("sp"), "margin"] = 4.0   # the Spanish passages move further
    whole, es = figures.delta_limits(df), df[df["stimulus_id"].str.startswith("sp")]
    assert figures.delta_limits(es) != pytest.approx(whole)          # the subsets really do differ
    fig = figures.margin_change(es, ylim=whole)
    assert fig.axes[0].get_ylim() == pytest.approx(whole)            # not the Spanish subset's own range


def test_moves_draws_a_figure_per_question_pooled_and_per_language(tmp_path):
    """One panel per intervention kind; each question drawn with both languages, es-only and
    fr-only, all on the whole run's axes. Identity is the start of every line, never a panel."""
    _m3_run(tmp_path)
    written = figures.m3_moves(tmp_path, formats=("png",))
    assert _files(written) == sorted(f"margin_moves_{q}{tag}.png" for q in ("anomaly", "report")
                                     for tag in ("", "_es", "_fr"))

    rows = figures.move_rows(figures.read_records(tmp_path))
    assert set(rows["kind"]) == {"swap", "label_to_present", "big_nonlabel", "random_direction"}
    assert (rows["clean_margin"] != rows["margin"]).any()        # a line with two distinct ends
    pytest.importorskip("scipy.stats")
    fig = figures.margin_moves(figures.read_records(tmp_path), "anomaly")
    titles = [ax.get_title() for ax in fig.axes]
    assert [t.split("\n")[0] for t in titles] == \
        [figures.KIND_LABELS[k] for k in ("swap", "label_to_present", "big_nonlabel", "random_direction")]
    # one p per panel, over the lines drawn there: swap has one m2i cell per passage (2), the
    # synthetic label_to_present one token x 2 directions x 2 passages (4)
    assert titles[0].split("\n")[1].startswith("p ") and titles[0].endswith("(n = 2)")
    assert titles[1].endswith("(n = 4)")


def test_presentation_figures_count_an_edit_made_the_same_in_both_directions_once(tmp_path):
    d = figures._one_edit_each(figures.read_records(_m3_run(tmp_path)))
    assert len(d[d["kind"] == "swap"]) == 4 and set(d.loc[d["kind"] == "swap", "direction"]) == {"m2i"}
    assert len(d[d["kind"] == "label_to_present"]) == 8      # removes a different label in each direction


def test_combined_summary_figures(tmp_path):
    a, b = _m3_run(tmp_path / "a"), _m3_run(tmp_path / "b")
    paths = figures.combined_summary_figures([a, b], formats=("png",), out_dir=tmp_path / "out")
    assert _files(paths) == ["flip_heatmap_combined.png", "margin_change_combined.png"]
    with pytest.raises(ValueError, match="pair_name"):
        figures.combined_summary_figures([a, _m3_run(tmp_path / "c", pair="plain")], out_dir=tmp_path / "out2")


def test_combined_panel_c_refuses_runs_that_differ(tmp_path):
    a = _m3_run(tmp_path / "a")
    b = _m3_run(tmp_path / "b")
    paths, info = figures.combined_panel_c([a, b], formats=("png",), out_dir=tmp_path / "out")
    assert [p.name for p in paths] == ["panel_c_combined.png"] and set(info) == {"a", "b"}
    c = _m3_run(tmp_path / "c", pair="plain")
    with pytest.raises(ValueError, match="pair_name"):
        figures.combined_panel_c([a, c], out_dir=tmp_path / "out2")
    m = _m3_run(tmp_path / "m", position_set=("instruction", "question", "matrix", "intrusion"))
    with pytest.raises(ValueError, match="position_set"):
        figures.combined_panel_c([a, m], out_dir=tmp_path / "out3")


def test_missing_inputs_are_skipped_not_fatal(tmp_path, capsys):
    assert figures.make_figures(tmp_path, milestone="M3") == []
    assert "records not found" in capsys.readouterr().err


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


def test_chinese_characters_are_drawn_not_boxes(tmp_path):
    """The positive-control prompt must render: no 'Glyph ... missing' warning in PNG, PDF or SVG."""
    import warnings

    plt = figures._plt()
    for fmt in ("png", "pdf", "svg"):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig, ax = plt.subplots()
            ax.text(0.1, 0.5, '·Spanish "小"的反义词是" 大 长')
            figures.write(fig, tmp_path / f"cjk.{fmt}")
        assert not [x for x in w if "missing from font" in str(x.message)], fmt
