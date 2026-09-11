import hashlib
import json
import threading
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

from jlens_spec import io as io_mod


@dataclass
class _Rec:
    a: int
    b: str
    arr: np.ndarray
    s: set


def test_append_records_roundtrip(tmp_path):
    path = tmp_path / "out.parquet"
    r1 = _Rec(a=1, b="x", arr=np.array([1.0, 2.0], dtype=np.float32), s={"p", "q"})
    io_mod.append_records([r1], path)
    r2 = _Rec(a=2, b="y", arr=np.array([3.0], dtype=np.float32), s={"z"})
    io_mod.append_records([r2], path)

    df = pd.read_parquet(path)
    assert len(df) == 2
    assert set(df["a"]) == {1, 2}
    assert set(df["b"]) == {"x", "y"}


def test_write_manifest_fields(tmp_path):
    io_mod.write_manifest(tmp_path, foo="bar", n=3)
    with open(tmp_path / "manifest.json") as f:
        manifest = json.load(f)
    assert manifest["foo"] == "bar"
    assert manifest["n"] == 3
    assert "written_at" in manifest


def test_sha256_of(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("hello")
    expected = hashlib.sha256(b"hello").hexdigest()
    assert io_mod.sha256_of(p) == expected


def test_background_uploader_never_runs_two_at_once(tmp_path, monkeypatch):
    calls = []
    in_flight = threading.Event()
    max_concurrent = [0]
    concurrent_now = [0]
    lock = threading.Lock()

    def fake_upload_run(run_dir):
        with lock:
            concurrent_now[0] += 1
            max_concurrent[0] = max(max_concurrent[0], concurrent_now[0])
        in_flight.set()
        time.sleep(0.1)
        calls.append(run_dir)
        with lock:
            concurrent_now[0] -= 1
        return f"url-for-{run_dir}"

    monkeypatch.setattr(io_mod, "upload_run", fake_upload_run)

    uploader = io_mod.BackgroundUploader("runs/_test", min_interval_s=0)
    uploader.trigger()
    in_flight.wait(timeout=2)
    # Fire several more triggers while the first upload is still running -- these should coalesce
    # into at most one more call, never run concurrently with the first.
    for _ in range(5):
        uploader.trigger()
    result, error = uploader.flush(timeout=5)
    uploader.shutdown()

    assert error is None
    assert result == "url-for-runs/_test"
    assert max_concurrent[0] == 1
    assert 1 <= len(calls) <= 2  # the first call, plus at most one coalesced follow-up


def test_background_uploader_surfaces_error_without_raising(tmp_path, monkeypatch):
    def failing_upload_run(run_dir):
        raise RuntimeError("boom")

    monkeypatch.setattr(io_mod, "upload_run", failing_upload_run)

    uploader = io_mod.BackgroundUploader("runs/_test", min_interval_s=0)
    uploader.trigger()
    result, error = uploader.flush(timeout=5)
    uploader.shutdown()

    assert result is None
    assert isinstance(error, RuntimeError)
    assert uploader.n_failures == 1


def test_upload_run_rejects_absolute_path(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        io_mod.upload_run(tmp_path / "runs" / "M0")


def test_append_records_leaves_no_temp_file_in_run_dir(tmp_path):
    path = tmp_path / "runs" / "M9" / "out.parquet"
    io_mod.append_records([{"a": 1}], path)
    io_mod.append_records([{"a": 2}], path)
    assert sorted(p.name for p in path.parent.iterdir()) == ["out.parquet"]
    assert list(pd.read_parquet(path)["a"]) == [1, 2]


class _Done:
    returncode = 0
    stdout = "\x1b[32m✓ Uploaded\x1b[0m\n  url: https://huggingface.co/datasets/x/y/commit/abc123\n"
    stderr = ""


def test_upload_run_returns_bare_url_and_excludes_figures_only_when_asked(monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return _Done()

    monkeypatch.setattr(io_mod.subprocess, "run", fake_run)
    assert io_mod.upload_run("runs/M0/smoke_1") == "https://huggingface.co/datasets/x/y/commit/abc123"
    assert "--exclude" not in calls[-1] and "--delete" not in calls[-1]
    io_mod.upload_run("runs/M0/smoke_1", exclude_figures=True)
    assert calls[-1][-2:] == ["--exclude", "figures/*"] and "--delete" not in calls[-1]


def _fake_uploads(tmp_path, fail_on=None):
    """Replacement for upload_run that records each call and which files existed at that moment.
    fail_on: "folder" or "summary" makes that upload raise."""
    calls = []

    def fake(path, exclude_figures=False):
        kind = "summary" if Path(path).name == "summary.md" else "folder"
        calls.append({"kind": kind, "exclude_figures": exclude_figures,
                      "files": sorted(p.name for p in tmp_path.iterdir())})
        if kind == fail_on:
            raise RuntimeError(f"{kind} upload broke")
        return f"https://example/{kind}-commit"

    return fake, calls


def test_finalize_run_uploads_folder_first_and_summary_last(tmp_path, monkeypatch):
    (tmp_path / "data.parquet").write_text("x")
    fake, calls = _fake_uploads(tmp_path)
    monkeypatch.setattr(io_mod, "upload_run", fake)

    url = io_mod.finalize_run(tmp_path, ["# summary", "body"], exclude_figures=True)

    assert url == "https://example/folder-commit"
    assert [c["kind"] for c in calls] == ["folder", "summary"]
    assert "summary.md" not in calls[0]["files"]  # the summary cannot go up with the data
    assert calls[0]["exclude_figures"] is True
    assert "summary.md" in calls[1]["files"]      # written only after the folder upload succeeded
    text = (tmp_path / "summary.md").read_text()
    assert "body" in text and "https://example/folder-commit" in text
    assert io_mod.dataset_url(tmp_path) in text
    assert not (tmp_path / "upload_error.txt").exists() and not (tmp_path / "summary_not_uploaded.md").exists()


def test_failed_folder_upload_writes_no_summary(tmp_path, monkeypatch):
    fake, calls = _fake_uploads(tmp_path, fail_on="folder")
    monkeypatch.setattr(io_mod, "upload_run", fake)

    assert io_mod.finalize_run(tmp_path, ["# s"]) is None
    assert [c["kind"] for c in calls] == ["folder"]
    assert not (tmp_path / "summary.md").exists() and not (tmp_path / "summary_not_uploaded.md").exists()
    assert "folder upload broke" in (tmp_path / "upload_error.txt").read_text()


def test_failed_summary_upload_renames_the_summary(tmp_path, monkeypatch):
    fake, calls = _fake_uploads(tmp_path, fail_on="summary")
    monkeypatch.setattr(io_mod, "upload_run", fake)

    assert io_mod.finalize_run(tmp_path, ["# s", "body"]) is None
    assert not (tmp_path / "summary.md").exists()
    assert "body" in (tmp_path / "summary_not_uploaded.md").read_text()
    assert "summary upload broke" in (tmp_path / "upload_error.txt").read_text()


def test_leftovers_from_a_failed_attempt_are_never_uploaded(tmp_path, monkeypatch):
    for name in ("summary.md", "summary_not_uploaded.md", "upload_error.txt"):
        (tmp_path / name).write_text("old")
    fake, calls = _fake_uploads(tmp_path)
    monkeypatch.setattr(io_mod, "upload_run", fake)

    io_mod.finalize_run(tmp_path, ["# s"])
    assert not {"summary.md", "summary_not_uploaded.md", "upload_error.txt"} & set(calls[0]["files"])


def test_finalize_run_leaves_the_manifest_untouched(tmp_path, monkeypatch):
    io_mod.write_manifest(tmp_path, run_id="r1")
    before = (tmp_path / "manifest.json").read_bytes()
    fake, _ = _fake_uploads(tmp_path)
    monkeypatch.setattr(io_mod, "upload_run", fake)
    io_mod.finalize_run(tmp_path, ["# s"])
    assert (tmp_path / "manifest.json").read_bytes() == before


# ------------------------------------------------ local stand-in store, per-file uploads, fp16


def test_local_store_mirrors_repo_relative_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = Path("runs/M2/loading_1")
    (run / "loadings").mkdir(parents=True)
    (run / "loadings" / "sp_01_report.parquet").write_text("a")
    (run / "figures" / "png").mkdir(parents=True)
    (run / "figures" / "png" / "x.png").write_text("f")
    store = io_mod.LocalStore("store")

    store.upload_files(run, ["loadings/sp_01_report.parquet"])
    assert store.exists(run / "loadings" / "sp_01_report.parquet")
    assert store.list_files(run) == {"runs/M2/loading_1/loadings/sp_01_report.parquet"}
    store.upload_path(run, exclude_figures=True)
    assert not store.exists(run / "figures" / "png" / "x.png")      # figures are never uploaded
    assert store.list_dirs("runs/M2") == {"loading_1"}
    (run / "loadings" / "sp_01_report.parquet").unlink()
    store.download(run / "loadings" / "sp_01_report.parquet")
    assert (run / "loadings" / "sp_01_report.parquet").read_text() == "a"
    assert store.list_files("runs/M9") == set() and store.list_dirs("runs/M9") == set()


def test_finalize_run_never_uploads_figures_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = Path("runs/M2/loading_1")
    (run / "figures" / "png").mkdir(parents=True)
    (run / "figures" / "png" / "x.png").write_text("f")
    (run / "data.parquet").write_text("d")
    store = io_mod.LocalStore("store")
    assert io_mod.finalize_run(run, ["# s"], store=store) is not None
    assert store.list_files(run) == {f"{run.as_posix()}/data.parquet", f"{run.as_posix()}/summary.md"}


def test_finalize_run_uploads_exactly_the_figures_it_is_given(tmp_path, monkeypatch):
    """M3 passes its summary figures; its mask figures (and every other figure) stay local."""
    monkeypatch.chdir(tmp_path)
    run = Path("runs/M3/r")
    (run / "figures" / "png" / "masks").mkdir(parents=True)
    for f in ("figures/png/panel_c.png", "figures/png/masks/sp_01_report.png", "data.parquet"):
        (run / f).write_text("x")
    store = io_mod.LocalStore("store")
    assert io_mod.finalize_run(run, ["# s"], store=store, figures_to_upload=["figures/png/panel_c.png"]) is not None
    assert store.list_files(run) == {f"{run.as_posix()}/{f}" for f in
                                     ("data.parquet", "figures/png/panel_c.png", "summary.md")}


def test_background_uploader_uploads_exactly_the_files_given(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = Path("runs/M3/r")
    for name in ("records/a.parquet", "records/b.parquet", "details/a.parquet"):
        (run / name).parent.mkdir(parents=True, exist_ok=True)
        (run / name).write_text(name)
    store = io_mod.LocalStore("store")
    up = io_mod.BackgroundUploader(run, store=store)
    up.trigger(files=["records/a.parquet", "details/a.parquet"])
    up.flush(timeout=5)
    up.shutdown()
    assert store.list_files(run) == {f"{run.as_posix()}/records/a.parquet", f"{run.as_posix()}/details/a.parquet"}
    assert up.uploaded == {"records/a.parquet", "details/a.parquet"} and not up.pending()


def test_background_uploader_requeues_failed_files(tmp_path):
    class Flaky:
        calls = 0

        def upload_files(self, run_dir, files):
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise RuntimeError("rate limited")
            return "ok"

    up = io_mod.BackgroundUploader("runs/M3/r", store=Flaky())
    up.trigger(files=["records/a.parquet"])
    up.flush(timeout=5)
    assert up.n_failures == 1 and up.pending() == {"records/a.parquet"}  # kept for the next upload
    up.trigger(files=["records/b.parquet"])
    up.flush(timeout=5)
    up.shutdown()
    assert up.pending() == set() and up.uploaded == {"records/a.parquet", "records/b.parquet"}


def test_records_table_keeps_full_vocab_vectors_in_fp16(tmp_path):
    import pyarrow.parquet as pq

    rows = [{"k": i, "logprobs_fp16": np.arange(5, dtype=np.float16) + i, "topk": [{"token": "a", "logit": 1.0}]}
            for i in range(3)]
    path = io_mod.write_parquet(io_mod.records_table(rows), tmp_path / "records" / "p.parquet")
    t = pq.read_table(path)
    assert str(t.schema.field("logprobs_fp16").type.value_type) == "halffloat"
    df = pd.read_parquet(path)
    assert np.array_equal(np.asarray(df["logprobs_fp16"][2], dtype=np.float16), np.arange(5, dtype=np.float16) + 2)
    assert list(df["k"]) == [0, 1, 2] and sorted(p.name for p in path.parent.iterdir()) == ["p.parquet"]
