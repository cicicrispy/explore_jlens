#!/usr/bin/env bash
# Idempotent environment setup for jlens-spec. Exits non-zero on any failure.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

REPO_ROOT="$(pwd)"

# 1. venv
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
PYVER="$(python3 -c 'import sys; print(sys.version_info[:2])')"
python3 - <<'EOF'
import sys
assert sys.version_info >= (3, 11), f"Python >=3.11 required, got {sys.version_info}"
EOF

# 2. platform detection
if [ "$(uname -s)" = "Darwin" ]; then
  PLATFORM="mac"
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  PLATFORM="cuda"
else
  PLATFORM="cpu-linux"
fi
echo "Detected platform: $PLATFORM"

# 3. torch per platform, then editable install
case "$PLATFORM" in
  mac)
    pip install --upgrade pip
    pip install torch
    ;;
  cuda)
    pip install --upgrade pip
    # Default CUDA wheel; record what nvidia-smi reports for the summary.
    nvidia-smi --query-gpu=driver_version,name --format=csv,noheader || true
    pip install torch
    ;;
  cpu-linux)
    pip install --upgrade pip
    pip install torch
    ;;
esac
pip install -e .

# 4. hf CLI + skills, gh CLI
pip install -U huggingface_hub  # the `hf` CLI ships with the base package (the [cli] extra no longer exists)
if [ -d ".agents/skills/hf-cli" ]; then
  echo "hf-cli skill already installed; skipping 'hf skills add --claude'."
else
  hf skills add --claude || echo "WARN: hf skills add --claude failed (non-fatal)"
fi

if ! command -v gh >/dev/null 2>&1; then
  if [ "$PLATFORM" = "mac" ]; then
    brew install gh
  else
    sudo apt-get update && sudo apt-get install -y gh
  fi
fi

# 5. secrets: .env is read ONLY by Python (jlens_spec.env.bootstrap -> python-dotenv), never
# sourced into this shell. Values are never printed; only set/unset is checked.
python3 - <<'EOF'
import os, subprocess, sys
from jlens_spec import env
env.bootstrap()
if not os.environ.get("HF_TOKEN"):
    sys.exit("ERROR: HF_TOKEN is not set in .env (needed for model/lens download and dataset upload).")
if os.environ.get("GH_TOKEN"):
    subprocess.run(["gh", "auth", "setup-git"], check=True)
else:
    print("GH_TOKEN not set; skipping 'gh auth setup-git' (git access assumed already configured).")
EOF

# 6. HF_HOME per platform, from configs/paths.yaml
export HF_HOME="$(python3 - <<EOF
import yaml
with open("configs/paths.yaml") as f:
    cfg = yaml.safe_load(f)
platform = "$PLATFORM"
key = "mac" if platform == "mac" else "cuda"
home = cfg["hf_home"][key]
print(home.replace("\$REPO", "$REPO_ROOT"))
EOF
)"
mkdir -p "$HF_HOME"
echo "HF_HOME=$HF_HOME"

# 7. cuda-only: download model + lens
if [ "$PLATFORM" = "cuda" ]; then
  python3 scripts/download.py
fi

# 8. tests -- not on the GPU box (human's decision, 2026-09-11; a departure from the spec): the
# suite runs the stand-in model with a random lens, which the Mac covers, and the GPU itself is
# checked by M1's validation. Elsewhere, fetch the stand-in first, with a visible progress bar --
# pytest captures output, so a first-time ~1.2 GB download inside a test fixture otherwise looks
# like a hang.
if [ "$PLATFORM" = "cuda" ]; then
  echo "cuda: tests skipped (they run on the Mac; M1 validation checks the GPU)."
else
  python3 - <<'EOF'
import yaml
from jlens_spec import env
env.bootstrap()
from huggingface_hub import snapshot_download
cfg = yaml.safe_load(open("configs/model.yaml"))
print(f"Fetching stand-in {cfg['standin_hf_id']} into HF_HOME (skips files already cached)...")
snapshot_download(cfg["standin_hf_id"])
EOF
  pytest tests/ -q
fi

# 9. lock file
pip freeze > "requirements-$PLATFORM.lock"
echo "SETUP OK ($PLATFORM)"
