import hashlib
import json
import threading
import time
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
