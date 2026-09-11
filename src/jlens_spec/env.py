"""Environment helpers: device selection, secret hygiene, provenance hashing."""
from __future__ import annotations

import hashlib
import os
import platform
import subprocess
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]


def bootstrap() -> None:
    """Call first in every script, BEFORE importing anything that pulls in huggingface_hub
    (jlens_spec.model, .lens, nnsight, transformers): huggingface_hub reads HF_HOME at import.

    - chdir to the repo root, so every path in the codebase is relative to the repo and scripts
      work from any cwd.
    - load secrets from .env into this process's environment (python-dotenv; values are never
      printed or logged, and variables already set are not overridden). This is the ONLY place
      .env is read -- never `source` it in a shell.
    - set HF_HOME from configs/paths.yaml (if not already set), so a fresh shell uses the same
      cache setup.sh downloaded into instead of silently re-downloading to ~/.cache.
    """
    os.chdir(_REPO_ROOT)
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env", override=False)
    if not os.environ.get("HF_HOME"):
        import yaml

        with open("configs/paths.yaml") as f:
            cfg = yaml.safe_load(f)
        key = "mac" if platform.system() == "Darwin" else "cuda"
        os.environ["HF_HOME"] = cfg["hf_home"][key]


def get_device() -> torch.device:
    """cuda if available, else cpu. Never returns mps -- tests must run on a fixed, unaccelerated
    backend for numerical reproducibility; MPS is opted into explicitly elsewhere, never by default."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def require_env(name: str) -> None:
    """Raise if the named environment variable is unset or empty. Never returns or logs its value."""
    if not os.environ.get(name):
        raise RuntimeError(f"required environment variable {name!r} is not set")


def git_commit() -> str:
    """Current commit hash, suffixed '-dirty' if any code/config differs from it (tracked edits or
    untracked non-ignored files, excluding runs/, which scripts write into). Records carrying a
    '-dirty' hash were NOT produced by the code at that commit -- commit before running milestones.
    Returns 'unknown' outside a git repo."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", ".", ":(exclude)runs"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if status else sha
    except Exception:
        return "unknown"


def config_hash(*paths: str) -> str:
    """sha256 hex digest of the concatenated bytes of every path given, in the order given."""
    h = hashlib.sha256()
    for p in paths:
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def expand_band(band_spec) -> list[int]:
    """Flatten a configs/bands.yaml band into a sorted, deduplicated list of layer indices.

    Every pair is INCLUSIVE on both ends: [18, 30] means layers 18..30 (13 layers). Accepts either a
    single pair (e.g. `workspace: [18, 50]`) or a list of pairs (e.g. `early_late: [[18, 30],
    [40, 50]]`). Downstream, `interventions.apply` edits every layer in the flat list inside one
    trace, whether it's one contiguous run or several blocks. `apply` separately refuses any layer
    the lens doesn't cover, which includes the final layer.
    """
    if not band_spec:
        raise ValueError("band_spec is null/empty -- the human has not filled this band in configs/bands.yaml yet")
    pairs = band_spec if isinstance(band_spec[0], (list, tuple)) else [band_spec]
    layers: set[int] = set()
    for start, end in pairs:
        if end < start:
            raise ValueError(f"band [{start}, {end}] has end < start")
        layers.update(range(start, end + 1))
    return sorted(layers)


BAND_NAMES = ("workspace", "full", "early_late")


def check_bands(bands: dict, lens_layers=None) -> dict[str, list[int]]:
    """M2 and M3 refuse to run unless configs/bands.yaml is complete and consistent. Returns
    {band name: expanded layer list}; raises ValueError listing every problem otherwise:
      - all three bands (workspace, full, early_late) are filled;
      - `full` starts at the same layer as `workspace` (the workspace onset);
      - `workspace` lies inside `full`;
      - every `early_late` layer lies inside `workspace`;
      - if `lens_layers` is given, every layer of every band is covered by the lens (so the final
        layer, which the lens never covers, can't be in a band).
    Together these keep sensory layers (below the onset) out of every band."""
    missing = [n for n in BAND_NAMES if not bands.get(n)]
    if missing:
        raise ValueError(f"configs/bands.yaml: {', '.join(missing)} not filled yet -- M2/M3 need all three "
                         "(read them off M1's band-signature and CKA figures)")
    ex = {n: expand_band(bands[n]) for n in BAND_NAMES}
    problems = []
    if ex["full"][0] != ex["workspace"][0]:
        problems.append(f"full starts at layer {ex['full'][0]}, workspace at {ex['workspace'][0]} "
                        "(full must start at the workspace onset)")
    if not set(ex["workspace"]) <= set(ex["full"]):
        problems.append(f"workspace layers {sorted(set(ex['workspace']) - set(ex['full']))} are not in full")
    if not set(ex["early_late"]) <= set(ex["workspace"]):
        problems.append(f"early_late layers {sorted(set(ex['early_late']) - set(ex['workspace']))} are not in workspace")
    if lens_layers is not None:
        uncovered = sorted({l for layers in ex.values() for l in layers} - set(lens_layers))
        if uncovered:
            problems.append(f"layers {uncovered} are not covered by the lens (the final layer never is)")
    if problems:
        raise ValueError("configs/bands.yaml is inconsistent: " + "; ".join(problems))
    return ex
