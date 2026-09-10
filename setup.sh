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
pip install -U "huggingface_hub[cli]"
hf skills add --claude || echo "WARN: hf skills add --claude failed (non-fatal)"

if ! command -v gh >/dev/null 2>&1; then
  if [ "$PLATFORM" = "mac" ]; then
    brew install gh
  else
    sudo apt-get update && sudo apt-get install -y gh
  fi
fi

# 5. secrets
if [ -f ".env" ]; then
  set -a
  source .env
  set +a
else
  echo "WARN: no .env found at repo root; assuming secrets are already present in the environment."
fi

if [ -n "${GH_TOKEN:-}" ]; then
  gh auth setup-git
else
  echo "GH_TOKEN not set; skipping 'gh auth setup-git' (assuming git/GitHub access is already configured for this machine)."
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set (needed for model/lens download and dataset upload)." >&2
  exit 1
fi

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

# 8. tests
pytest tests/ -q
if [ "$PLATFORM" = "cuda" ]; then
  pytest tests/ -q -m gpu
fi

# 9. lock file
pip freeze > "requirements-$PLATFORM.lock"
echo "SETUP OK ($PLATFORM)"
