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


def test_name_suffix_goes_between_name_and_time(tmp_path):
    exp, files = _setup(tmp_path)
    run = runs.start_run("M0", exp, files, root=tmp_path / "runs", name_suffix="Qwen3.6-27B")
    assert re.fullmatch(r"smoke_Qwen3\.6-27B_\d{8}-\d{6}", run.run_id)


# ------------------------------------------------------------------------------------ resume


def _resume_setup(tmp_path, monkeypatch):
    from jlens_spec import io as io_mod

    monkeypatch.chdir(tmp_path)
    exp = tmp_path / "exp_loading.yaml"
    exp.write_text(yaml.safe_dump({"name": "loading", "milestone": "M2"}))
    model = tmp_path / "model.yaml"
    model.write_text(yaml.safe_dump({"hf_id": "a/b"}))
    return exp, [model], io_mod.LocalStore("store")


def _finish(run, store):
    (run.dir / "summary.md").write_text("done")
    store.upload_path(run.dir)


def test_first_start_makes_a_new_run(tmp_path, monkeypatch):
    exp, files, store = _resume_setup(tmp_path, monkeypatch)
    run, resumed = runs.start_or_resume("M2", exp, files, store, root="runs")
    assert not resumed and run.dir.parent.as_posix() == "runs/M2" and run.dir.name.startswith("loading_")


def test_latest_unfinished_run_is_resumed_and_a_finished_one_is_not(tmp_path, monkeypatch):
    exp, files, store = _resume_setup(tmp_path, monkeypatch)
    a, _ = runs.start_or_resume("M2", exp, files, store, root="runs")
    b, resumed = runs.start_or_resume("M2", exp, files, store, root="runs")
    assert resumed and b.dir == a.dir                      # unfinished -> resumed
    _finish(a, store)
    c, resumed = runs.start_or_resume("M2", exp, files, store, root="runs")
    assert not resumed and c.dir != a.dir                  # finished -> a new run


def test_an_older_unfinished_run_is_never_picked_up_by_default(tmp_path, monkeypatch):
    exp, files, store = _resume_setup(tmp_path, monkeypatch)
    old, _ = runs.start_or_resume("M2", exp, files, store, root="runs")        # abandoned, unfinished
    newer, _ = runs.start_or_resume("M2", exp, files, store, root="runs", fresh=True)
    _finish(newer, store)
    nxt, resumed = runs.start_or_resume("M2", exp, files, store, root="runs")
    assert not resumed and nxt.dir not in (old.dir, newer.dir)
    again, resumed = runs.start_or_resume("M2", exp, files, store, root="runs", resume=old.dir)
    assert resumed and again.dir == old.dir                # only when named explicitly


def test_after_a_code_change_the_next_start_is_a_new_run(tmp_path, monkeypatch, capsys):
    exp, files, store = _resume_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(runs.env, "git_commit", lambda: "aaa")
    a, _ = runs.start_or_resume("M2", exp, files, store, root="runs")
    monkeypatch.setattr(runs.env, "git_commit", lambda: "bbb")
    b, resumed = runs.start_or_resume("M2", exp, files, store, root="runs")
    assert not resumed and b.dir != a.dir and "other code" in capsys.readouterr().out
    c, resumed = runs.start_or_resume("M2", exp, files, store, root="runs", resume=a.dir)
    assert resumed and c.dir == a.dir and "other code" in capsys.readouterr().out  # --resume still does, and says so


def test_a_finished_run_is_never_resumed(tmp_path, monkeypatch):
    exp, files, store = _resume_setup(tmp_path, monkeypatch)
    a, _ = runs.start_or_resume("M2", exp, files, store, root="runs")
    _finish(a, store)
    with pytest.raises(SystemExit, match="already finished"):
        runs.start_or_resume("M2", exp, files, store, root="runs", resume=a.dir)


def test_resume_keeps_its_settings_copy_and_reports_changes(tmp_path, monkeypatch, capsys):
    exp, files, store = _resume_setup(tmp_path, monkeypatch)
    a, _ = runs.start_or_resume("M2", exp, files, store, root="runs")
    files[0].write_text(yaml.safe_dump({"hf_id": "c/d"}))
    b, resumed = runs.start_or_resume("M2", exp, files, store, root="runs")
    assert resumed and b.load("model.yaml")["hf_id"] == "a/b"
    assert runs.settings_diff(b) == ["model.yaml"]
    assert "model.yaml" in capsys.readouterr().out


def test_resume_on_a_new_machine_fetches_the_run_from_the_store(tmp_path, monkeypatch):
    import shutil

    exp, files, store = _resume_setup(tmp_path, monkeypatch)
    a, _ = runs.start_or_resume("M2", exp, files, store, root="runs")
    (a.dir / "loadings").mkdir()
    (a.dir / "loadings" / "sp_01_report.parquet").write_text("x")
    store.upload_path(a.dir)
    shutil.rmtree("runs")                                  # a new machine: nothing local
    b, resumed = runs.start_or_resume("M2", exp, files, store, root="runs")
    assert resumed and b.run_id == a.run_id and (b.dir / "settings" / "model.yaml").exists()
    assert runs.uploaded_files(b.dir, store) >= {"loadings/sp_01_report.parquet", "manifest.json"}
    assert not (b.dir / "loadings" / "sp_01_report.parquet").exists()
    assert runs.sync_down(b.dir, store) == ["loadings/sp_01_report.parquet"]
    assert (b.dir / "loadings" / "sp_01_report.parquet").read_text() == "x"
