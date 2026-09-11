"""Run folders (runs.py): each run gets its own folder with a copy of its settings, and reads
settings only from that copy -- so editing configs/ after a run starts never changes that run."""
import json
import re

import pytest
import yaml

from jlens_spec import runs


def _setup(tmp_path, milestone="M0"):
    exp = tmp_path / "exp_smoke.yaml"
    exp.write_text(yaml.safe_dump({"name": "smoke", "milestone": milestone, "save_topk": 100}))
    model = tmp_path / "model.yaml"
    model.write_text(yaml.safe_dump({"hf_id": "a/b", "revision": "main"}))
    stim = tmp_path / "stimuli.json"
    stim.write_text(json.dumps({"passages": [], "questions": {}}))
    return exp, [model, stim]


def test_start_run_creates_named_folder_with_settings_copy(tmp_path):
    exp, files = _setup(tmp_path)
    run = runs.start_run("M0", exp, files, root=tmp_path / "runs")

    assert run.dir.parent == tmp_path / "runs" / "M0"
    assert re.fullmatch(r"smoke_\d{8}-\d{6}", run.run_id)
    assert run.dir.name == run.run_id
    for src in [exp, *files]:
        assert (run.dir / "settings" / src.name).read_bytes() == src.read_bytes()
    m = run.manifest()
    assert m["milestone"] == "M0" and m["run_id"] == run.run_id
    assert "status" not in m  # finished = summary.md on HF; the manifest never changes
    assert m["experiment_file"] == "exp_smoke.yaml"
    assert run.experiment["save_topk"] == 100
    assert run.load("stimuli.json") == {"passages": [], "questions": {}}


def test_run_reads_its_copy_not_the_edited_source(tmp_path):
    exp, files = _setup(tmp_path)
    run = runs.start_run("M0", exp, files, root=tmp_path / "runs")
    hash_before = run.config_hash()

    exp.write_text(yaml.safe_dump({"name": "smoke", "milestone": "M0", "save_topk": 20}))
    files[0].write_text(yaml.safe_dump({"hf_id": "c/d", "revision": "other"}))

    reopened = runs.open_run(run.dir)
    assert reopened.experiment["save_topk"] == 100
    assert reopened.load("model.yaml")["hf_id"] == "a/b"
    assert reopened.config_hash() == hash_before


def test_config_hash_covers_every_settings_file(tmp_path):
    exp, files = _setup(tmp_path)
    run = runs.start_run("M0", exp, files, root=tmp_path / "runs")
    before = run.config_hash()
    (run.dir / "settings" / "stimuli.json").write_text("{}")
    assert run.config_hash() != before


def test_runs_never_share_a_folder(tmp_path):
    exp, files = _setup(tmp_path)
    a = runs.start_run("M0", exp, files, root=tmp_path / "runs")
    b = runs.start_run("M0", exp, files, root=tmp_path / "runs")  # same second
    assert a.dir != b.dir and a.dir.exists() and b.dir.exists()


def test_experiment_for_another_milestone_is_rejected(tmp_path):
    exp, files = _setup(tmp_path, milestone="M3")
    with pytest.raises(ValueError, match="milestone 'M3', not 'M0'"):
        runs.start_run("M0", exp, files, root=tmp_path / "runs")


def test_two_settings_files_with_the_same_name_are_rejected(tmp_path):
    exp, files = _setup(tmp_path)
    other = tmp_path / "sub"
    other.mkdir()
    (other / "model.yaml").write_text("x: 1")
    with pytest.raises(ValueError, match="both named 'model.yaml'"):
        runs.start_run("M0", exp, [*files, other / "model.yaml"], root=tmp_path / "runs")
