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


def write_parquet(data, path) -> Path:
    """Write one complete parquet file atomically (temp file one directory up, then os.replace --
    same reasoning as `append_records`). `data`: a DataFrame, a pyarrow Table, or a list of
    records (dataclasses or dicts). Used for the per-prompt checkpoint files, each written once."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, list):
        data = pd.DataFrame([_sanitize(r) for r in data])
    table = data if isinstance(data, pa.Table) else pa.Table.from_pandas(data, preserve_index=False)
    tmp = path.parent.parent / f".{path.parent.name}_{path.name}.{os.getpid()}.tmp"
    pq.write_table(table, tmp)
    os.replace(tmp, path)
    return path


def records_table(records, fp16_cols=("logprobs_fp16",)):
    """Records (dataclasses or dicts) -> pyarrow Table, keeping each column in `fp16_cols` (a
    full-vocabulary vector per row) as fixed-size lists of 16-bit floats. `_sanitize` would turn
    them into Python floats, stored as 64-bit -- 4x the size (~2 MB per 248k-token vector)."""
    import pyarrow as pa

    rows = [_sanitize({k: v for k, v in (dataclasses.asdict(r) if dataclasses.is_dataclass(r) else r).items()
                       if k not in fp16_cols}) for r in records]
    table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
    for col in fp16_cols:
        arrays = [np.asarray((dataclasses.asdict(r) if dataclasses.is_dataclass(r) else r)[col], dtype=np.float16)
                  for r in records]
        if not arrays:
            continue
        size = arrays[0].shape[0]
        assert all(a.shape == (size,) for a in arrays), f"{col}: every row must be a vector of the same length"
        table = table.append_column(col, pa.FixedSizeListArray.from_arrays(pa.array(np.concatenate(arrays)), size))
    return table


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


class HFStore:
    """The HF dataset `orbitsoferis/jlens-specificity` -- the only place a run counts as finished,
    and the only backup of runs/ (which git ignores). All paths are repo-relative and identical
    inside the dataset (e.g. "runs/M2/loading_20260912-031000/loadings/sp_01_report.parquet")."""

    name = f"HF dataset {HF_DATASET}"

    def folder_url(self, path) -> str:
        return dataset_url(path)

    def upload_path(self, path, exclude_figures: bool = False) -> str:
        """One file or a whole folder (`hf upload`, see upload_run). Returns the commit URL."""
        return upload_run(path, exclude_figures=exclude_figures)

    def upload_files(self, run_dir, rel_files) -> str:
        """Exactly these files (paths relative to `run_dir`), in ONE commit, nothing else from the
        folder. Returns the commit URL."""
        from huggingface_hub import HfApi

        run_dir = Path(run_dir).as_posix()
        info = HfApi().upload_folder(repo_id=HF_DATASET, repo_type="dataset", folder_path=run_dir,
                                     path_in_repo=run_dir, allow_patterns=sorted(rel_files),
                                     commit_message=f"{run_dir}: {len(rel_files)} file(s)")
        return info.commit_url

    def list_files(self, prefix) -> set[str]:
        """Every file under `prefix` (repo-relative paths); empty if `prefix` isn't in the dataset."""
        from huggingface_hub import HfApi
        from huggingface_hub.errors import EntryNotFoundError
        from huggingface_hub.hf_api import RepoFile

        try:
            return {e.path for e in HfApi().list_repo_tree(HF_DATASET, path_in_repo=Path(prefix).as_posix(),
                                                           recursive=True, repo_type="dataset")
                    if isinstance(e, RepoFile)}
        except EntryNotFoundError:
            return set()

    def list_dirs(self, prefix) -> set[str]:
        """Names of the folders directly under `prefix`; empty if `prefix` isn't in the dataset."""
        from huggingface_hub import HfApi
        from huggingface_hub.errors import EntryNotFoundError
        from huggingface_hub.hf_api import RepoFolder

        try:
            return {Path(e.path).name for e in HfApi().list_repo_tree(
                HF_DATASET, path_in_repo=Path(prefix).as_posix(), repo_type="dataset")
                if isinstance(e, RepoFolder)}
        except EntryNotFoundError:
            return set()

    def exists(self, path) -> bool:
        from huggingface_hub import HfApi

        return HfApi().file_exists(HF_DATASET, Path(path).as_posix(), repo_type="dataset")

    def download(self, path) -> Path:
        """Fetch one file to the same repo-relative path on this machine (via the HF cache)."""
        import shutil

        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(HF_DATASET, Path(path).as_posix(), repo_type="dataset")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, path)
        return Path(path)


class LocalStore:
    """A local folder standing in for the HF dataset, with the same repo-relative layout under
    `root` -- used by the Mac dry run and by the tests, so they never touch the real dataset.
    Nothing here goes over the network."""

    def __init__(self, root):
        self.root = Path(root)
        self.name = f"local stand-in for the HF dataset ({self.root})"

    def folder_url(self, path) -> str:
        return (self.root / path).as_posix()

    def _copy(self, src: Path) -> None:
        import shutil

        dst = self.root / src
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)

    def upload_path(self, path, exclude_figures: bool = False) -> str:
        path = Path(path)
        if path.is_absolute():
            raise ValueError(f"store paths must be repo-relative, got {path}")
        files = [path] if path.is_file() else [
            p for p in sorted(path.rglob("*")) if p.is_file()
            and not (exclude_figures and "figures" in p.relative_to(path).parts[:1])]
        for f in files:
            self._copy(f)
        return f"local://{(self.root / path).as_posix()}"

    def upload_files(self, run_dir, rel_files) -> str:
        for f in rel_files:
            self._copy(Path(run_dir) / f)
        return f"local://{(self.root / run_dir).as_posix()}"

    def list_files(self, prefix) -> set[str]:
        base = self.root / prefix
        if not base.exists():
            return set()
        return {p.relative_to(self.root).as_posix() for p in base.rglob("*") if p.is_file()}

    def list_dirs(self, prefix) -> set[str]:
        base = self.root / prefix
        return {p.name for p in base.iterdir() if p.is_dir()} if base.exists() else set()

    def exists(self, path) -> bool:
        return (self.root / path).is_file()

    def download(self, path) -> Path:
        import shutil

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.root / path, path)
        return Path(path)


def finalize_run(run_dir, summary_lines: list[str], exclude_figures: bool = True, store=None) -> str | None:
    """End of every milestone script. **A run is finished if and only if its summary.md is in the
    store** (the HF dataset; a local stand-in folder for dry runs).

    1. Upload the run folder -- WITHOUT figures/ (no milestone uploads figures; each is redrawn from
       the uploaded data by scripts/make_figures.py). summary.md does not exist yet, so it cannot go
       up early (hf may split the folder into several commits -- that's fine).
    2. Only after (1) has fully succeeded: write summary.md -- `summary_lines` plus the run's
       dataset folder link and (1)'s URL -- and upload it on its own, as the last step.

    Failures are recorded, never raised, and never leave a summary.md that isn't uploaded:
    - (1) fails: no summary is written; the error goes to upload_error.txt.
    - (2) fails: summary.md is renamed summary_not_uploaded.md; the error goes to upload_error.txt.
    Leftovers from an earlier attempt (summary.md, summary_not_uploaded.md, upload_error.txt) are
    deleted first, so they are never uploaded in step 1. Returns (1)'s URL, or None on failure."""
    store = store or HFStore()
    run_dir = Path(run_dir)
    summary = run_dir / "summary.md"
    not_uploaded = run_dir / "summary_not_uploaded.md"
    error_file = run_dir / "upload_error.txt"
    for leftover in (summary, not_uploaded, error_file):
        leftover.unlink(missing_ok=True)

    def fail(what: str, e: Exception) -> None:
        error_file.write_text(f"{what} failed: {e!r}\n(HF_TOKEN must be set; see .env)\n")
        print(f"[upload] {what} FAILED -- this run counts as unfinished (no summary.md in the store). "
              f"Error saved to {error_file}: {e!r}", file=sys.stderr, flush=True)

    try:
        url = store.upload_path(run_dir, exclude_figures=exclude_figures)
    except Exception as e:  # noqa: BLE001 -- recorded in upload_error.txt, not hidden
        fail("uploading the run folder", e)
        return None

    summary.write_text("\n".join(summary_lines) + "\n"
                       f"- Dataset folder ({store.name}): {store.folder_url(run_dir)}\n"
                       f"- Upload of the run folder: {url}\n")
    try:
        store.upload_path(summary)
    except Exception as e:  # noqa: BLE001
        summary.rename(not_uploaded)
        fail("uploading summary.md", e)
        return None
    print(f"[upload] done -- {store.folder_url(run_dir)}", flush=True)
    return url


class BackgroundUploader:
    """Uploads on a single background thread, so a GPU loop never blocks on the network --
    satisfies hygiene invariant #10 ("sync artifacts off the ephemeral instance as they are
    produced") without stalling M2/M3's per-prompt loop.

    Two ways to use it:
    - `trigger(files=[...])` (M2/M3): upload exactly these files -- paths relative to the run folder,
      e.g. each finished prompt's checkpoint files -- in one commit, nothing else from the folder.
      `uploaded` holds every file confirmed uploaded. A failed batch goes back into the queue and is
      retried with the next trigger (and finalize_run's folder upload picks up anything left).
    - `trigger()` with no files: upload the whole run folder (`upload_run`).

    At most one upload runs at a time. Calling `trigger()` while one is already in flight does not
    start a second (concurrent uploads to the same repo path would race); the files queue up and
    exactly one more upload runs right after the current one finishes, taking everything queued in
    the meantime. `min_interval_s` (default 0 = no delay) optionally spaces uploads apart if Hub
    rate limits bite; any wait happens on the upload thread, never the GPU loop.

    Uploads never raise into the caller's thread, but every failure is printed to stderr as it
    happens and counted in `n_failures`; `flush()` returns (last_result, last_error).
    """

    def __init__(self, run_dir, min_interval_s: float = 0.0, store=None):
        self.run_dir = str(run_dir)
        self.min_interval_s = min_interval_s
        self.store = store
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hf-upload")
        self._lock = threading.Lock()
        self._busy = False
        self._dirty = False
        self._pending: set[str] = set()
        self._whole_folder = False
        self._last_start = 0.0
        self.uploaded: set[str] = set()
        self.last_result: str | None = None
        self.last_error: Exception | None = None
        self.n_uploads = 0
        self.n_failures = 0

    def trigger(self, files=None) -> None:
        with self._lock:
            if files is None:
                self._whole_folder = True
            else:
                self._pending.update(str(f) for f in files)
            if self._busy:
                self._dirty = True
                return
            self._busy = True
        self._executor.submit(self._run_once)

    def _run_once(self) -> None:
        with self._lock:
            files, self._pending = self._pending, set()
            whole, self._whole_folder = self._whole_folder, False
        try:
            wait = self.min_interval_s - (time.time() - self._last_start)
            if wait > 0:
                time.sleep(wait)
            self._last_start = time.time()
            if whole:
                self.last_result = (upload_run(self.run_dir) if self.store is None
                                    else self.store.upload_path(self.run_dir))
            if files:
                self.last_result = (self.store or HFStore()).upload_files(self.run_dir, files)
                self.uploaded |= files
            self.last_error = None
            self.n_uploads += 1
        except Exception as e:  # noqa: BLE001 -- reported below, never raised into the GPU loop
            with self._lock:  # retried with the next trigger; finalize_run's folder upload is the backstop
                self._pending |= files
                self._whole_folder = self._whole_folder or whole
            self.last_error = e
            self.n_failures += 1
            print(f"[hf-upload] WARNING: background upload of {self.run_dir} failed "
                  f"({self.n_failures} so far; will retry with the next upload): {e}", file=sys.stderr, flush=True)
        finally:
            with self._lock:
                again = self._dirty
                self._dirty = False
                self._busy = again
            if again:
                self._executor.submit(self._run_once)

    def pending(self) -> set[str]:
        """Files queued but not yet confirmed uploaded (e.g. after a failure)."""
        with self._lock:
            return set(self._pending)

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
