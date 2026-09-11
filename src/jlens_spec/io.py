"""Artifact I/O: incremental parquet writes, manifests, HF dataset upload, hashing."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


def _sanitize(v):
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return {k: _sanitize(vv) for k, vv in dataclasses.asdict(v).items()}
    if isinstance(v, (set, frozenset)):
        return sorted(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_sanitize(x) for x in v]
    if isinstance(v, dict):
        return {k: _sanitize(x) for k, x in v.items()}
    return v


def append_records(records, path) -> None:
    """Append `records` (dataclasses or dicts) to a parquet file at `path`, read-modify-write.

    The new file is written to a temp file one directory UP from `path` (so a background
    `hf upload` of `path`'s run directory never sees it), then swapped in with os.replace, which is
    atomic on the same filesystem. A crash mid-write therefore leaves the previous complete file,
    and a concurrent reader sees either the old or the new file, never a half-written one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_sanitize(r) for r in records]
    new_df = pd.DataFrame(rows)
    if path.exists():
        old_df = pd.read_parquet(path)
        df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        df = new_df
    tmp = path.parent.parent / f".{path.parent.name}_{path.name}.{os.getpid()}.tmp"
    df.to_parquet(tmp)
    os.replace(tmp, path)


def write_manifest(run_dir, **fields) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(fields)
    manifest.setdefault("written_at", time.time())
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)


HF_DATASET = "orbitsoferis/jlens-specificity"


def dataset_url(path) -> str:
    """Browser link to a repo-relative run path (e.g. a run folder) inside the HF dataset."""
    return f"https://huggingface.co/datasets/{HF_DATASET}/tree/main/{Path(path).as_posix()}"


def upload_run(path, exclude_figures: bool = False) -> str:
    """hf upload orbitsoferis/jlens-specificity <path> <path> --repo-type dataset
    [--exclude "figures/*"]

    `path` is a run folder or a single file in one, repo-relative (e.g.
    "runs/M3/stage1_20260912-031000" or ".../summary.md"); it's used as-is for the path inside the
    dataset, so an absolute path would land at a machine-specific location. Run from the repo root
    (env.bootstrap() does this). `exclude_figures` leaves the run's figures/ folder out (M0: its mask
    figures are redrawn from masks.parquet whenever needed). Nothing already in the dataset is
    deleted. hf may split a large folder into several commits. Raises RuntimeError carrying hf's
    stderr on failure. Returns the URL hf prints (its colored "✓ Uploaded / url: ..." output reduced
    to the bare URL), or the de-colored output if no URL is found.
    """
    rel = Path(path)
    if rel.is_absolute():
        raise ValueError(f"upload_run needs a repo-relative path like 'runs/M3/<run_id>', got {path!r}")
    rel = rel.as_posix()
    cmd = ["hf", "upload", HF_DATASET, rel, rel, "--repo-type", "dataset"]
    if exclude_figures:
        cmd += ["--exclude", "figures/*"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"hf upload {rel} failed (exit {result.returncode}): {result.stderr.strip()[-2000:]}")
    out = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout).strip()  # hf colors its output even when piped
    url = re.search(r"https://\S+", out)
    return url.group(0) if url else out


def finalize_run(run_dir, summary_lines: list[str], exclude_figures: bool = False) -> str | None:
    """End of every milestone script. **A run is finished if and only if its summary.md is on HF.**

    1. Upload the run folder. summary.md does not exist yet, so it cannot go up early (hf may split
       the folder into several commits -- that's fine).
    2. Only after (1) has fully succeeded: write summary.md -- `summary_lines` plus the run's HF
       folder link and (1)'s URL -- and upload it on its own, as the last step.

    Failures are recorded, never raised, and never leave a summary.md that isn't on HF:
    - (1) fails: no summary is written; the error goes to upload_error.txt.
    - (2) fails: summary.md is renamed summary_not_uploaded.md; the error goes to upload_error.txt.
    Leftovers from an earlier attempt (summary.md, summary_not_uploaded.md, upload_error.txt) are
    deleted first, so they are never uploaded in step 1. Returns (1)'s URL, or None on failure."""
    run_dir = Path(run_dir)
    summary = run_dir / "summary.md"
    not_uploaded = run_dir / "summary_not_uploaded.md"
    error_file = run_dir / "upload_error.txt"
    for leftover in (summary, not_uploaded, error_file):
        leftover.unlink(missing_ok=True)

    def fail(what: str, e: Exception) -> None:
        error_file.write_text(f"{what} failed: {e!r}\n(HF_TOKEN must be set; see .env)\n")
        print(f"[upload] {what} FAILED -- this run counts as unfinished (no summary.md on HF). "
              f"Error saved to {error_file}: {e!r}", file=sys.stderr, flush=True)

    try:
        url = upload_run(run_dir, exclude_figures=exclude_figures)
    except Exception as e:  # noqa: BLE001 -- recorded in upload_error.txt, not hidden
        fail("uploading the run folder", e)
        return None

    summary.write_text("\n".join(summary_lines) + "\n"
                       f"- HF dataset folder: {dataset_url(run_dir)}\n"
                       f"- HF upload of the run folder: {url}\n")
    try:
        upload_run(summary)
    except Exception as e:  # noqa: BLE001
        summary.rename(not_uploaded)
        fail("uploading summary.md", e)
        return None
    print(f"[upload] done -- {dataset_url(run_dir)}", flush=True)
    return url


class BackgroundUploader:
    """Runs `upload_run` in a single background thread so a GPU loop writing records (via
    `append_records`) never blocks on the network `hf upload` call -- satisfies hygiene invariant
    #10 ("sync artifacts off the ephemeral instance as they are produced") without stalling M2/M3's
    per-cell trace loop.

    Usage: create one per run directory, call `.trigger()` after each `append_records` (or as often
    as you like -- it's cheap and self-throttling), and call `.flush()` once at the end of the
    script to block until the final upload has actually completed before printing the summary.

    At most one `hf upload` subprocess runs at a time. Calling `trigger()` while one is already in
    flight does not spawn a second (concurrent `hf upload` calls against the same repo path would
    race); instead it marks the uploader "dirty" so exactly one more upload runs immediately after
    the current one finishes, picking up everything written in the meantime. `min_interval_s`
    (default 0 = no delay) optionally spaces consecutive uploads apart if Hub rate limits bite; any
    wait happens on the upload thread, never the GPU loop.

    Uploads never raise into the caller's thread, but every failure is printed to stderr as it
    happens and counted in `n_failures`; `flush()` returns (last_result, last_error).
    """

    def __init__(self, run_dir, min_interval_s: float = 0.0):
        self.run_dir = str(run_dir)
        self.min_interval_s = min_interval_s
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hf-upload")
        self._lock = threading.Lock()
        self._busy = False
        self._dirty = False
        self._last_start = 0.0
        self.last_result: str | None = None
        self.last_error: Exception | None = None
        self.n_uploads = 0
        self.n_failures = 0

    def trigger(self) -> None:
        with self._lock:
            if self._busy:
                self._dirty = True
                return
            self._busy = True
        self._executor.submit(self._run_once)

    def _run_once(self) -> None:
        try:
            wait = self.min_interval_s - (time.time() - self._last_start)
            if wait > 0:
                time.sleep(wait)
            self._last_start = time.time()
            self.last_result = upload_run(self.run_dir)
            self.last_error = None
            self.n_uploads += 1
        except Exception as e:  # noqa: BLE001 -- reported below, never raised into the GPU loop
            self.last_error = e
            self.n_failures += 1
            print(f"[hf-upload] WARNING: background upload of {self.run_dir} failed "
                  f"({self.n_failures} so far): {e}", file=sys.stderr, flush=True)
        finally:
            with self._lock:
                again = self._dirty
                self._dirty = False
                self._busy = again
            if again:
                self._executor.submit(self._run_once)

    def flush(self, timeout: float | None = None):
        """Block until no upload is in flight or queued. Returns (last_result, last_error)."""
        start = time.time()
        while True:
            with self._lock:
                if not self._busy:
                    break
            if timeout is not None and time.time() - start > timeout:
                raise TimeoutError("background HF upload did not finish within the given timeout")
            time.sleep(0.5)
        return self.last_result, self.last_error

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
