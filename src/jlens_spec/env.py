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
    - set HF_HOME from configs/paths.yaml (if not already set in the shell), so a fresh shell uses
      the same cache setup.sh downloaded into instead of silently re-downloading to ~/.cache.
    Secrets are NOT loaded here: source .env in your shell (see README).
    """
    os.chdir(_REPO_ROOT)
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
