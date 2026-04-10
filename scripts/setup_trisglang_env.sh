#!/usr/bin/env bash
# Setup the 'trisglang' conda environment for TriAttention sglang integration.
#
# Prerequisites:
#   - conda (Miniconda or Anaconda)
#   - CUDA toolkit 12.x installed and CUDA_HOME set
#
# Usage:
#   bash scripts/setup_trisglang_env.sh
#
# After running:
#   conda activate trisglang

set -euo pipefail

ENV_NAME="trisglang"
PYTHON_VERSION="3.10"
SGLANG_VERSION="0.5.10"

echo "=== Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION} ==="
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

# Activate the environment within this script
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "=== Upgrading pip ==="
pip install --upgrade pip

echo "=== Installing sglang v${SGLANG_VERSION} ==="
# Install sglang from PyPI. This pulls in torch, flashinfer, and other deps.
# If uv is available, use it for faster installs; otherwise fall back to pip.
if command -v uv &>/dev/null; then
    uv pip install "sglang==${SGLANG_VERSION}"
else
    pip install "sglang==${SGLANG_VERSION}"
fi

echo "=== Installing triattention package (editable) ==="
# Assumes this script is run from the dc1-release repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
pip install -e "${REPO_ROOT}"

echo "=== Verifying installation ==="
python -c "import sglang; print(f'sglang version: {sglang.__version__}')"
python -c "from triattention.sglang import install_sglang_integration; print('triattention.sglang importable: OK')"

echo ""
echo "=== Setup complete ==="
echo "Activate with:  conda activate ${ENV_NAME}"
echo "Launch server:  python -m triattention.sglang --model <model_path>"
