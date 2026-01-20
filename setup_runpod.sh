#!/bin/bash
# RunPod setup script for Hybridyzer
# Usage: bash setup_runpod.sh
#
# Run this after cloning to /workspace

set -e

echo "=========================================="
echo "  Hybridyzer RunPod Setup"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "train.py" ]; then
    echo "Error: Run this script from the Hybridyzer root directory"
    echo "  cd /workspace/Hybridyzer && bash setup_runpod.sh"
    exit 1
fi

# Prefer conda env for isolation when available
ENV_FILE="environment.runpod.yml"
ENV_NAME="$(awk -F': ' '/^name:/ {print $2}' "$ENV_FILE" 2>/dev/null || echo "hybridyzer")"
CONDA_DIR="${CONDA_DIR:-}"

if command -v conda &>/dev/null; then
    echo "[1/4] Conda detected. Using isolated env: $ENV_NAME"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        conda env update -f "$ENV_FILE"
    else
        conda env create -f "$ENV_FILE"
    fi
    conda activate "$ENV_NAME"
    echo "[1/4] Installing extra dependencies in conda env..."
    pip install joblib scipy numba
else
    echo "[1/4] Conda not found. Installing Miniconda for isolation..."
    if [ -z "$CONDA_DIR" ]; then
        if [ -w /opt ]; then
            CONDA_DIR="/opt/conda"
        else
            CONDA_DIR="$HOME/miniconda3"
        fi
    fi
    if [ ! -x "$CONDA_DIR/bin/conda" ]; then
        curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
        bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    fi
    export PATH="$CONDA_DIR/bin:$PATH"
    source "$CONDA_DIR/etc/profile.d/conda.sh"
    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        conda env update -f "$ENV_FILE"
    else
        conda env create -f "$ENV_FILE"
    fi
    conda activate "$ENV_NAME"
    echo "[1/4] Installing extra dependencies in conda env..."
    pip install joblib scipy numba
fi

# Install RAPIDS for GPU acceleration (pip-only fallback)
echo ""
echo "[2/4] Installing RAPIDS GPU libraries..."
if python -c "import cuml, cudf" &>/dev/null; then
    echo "RAPIDS already available in conda env"
else
    pip install cudf-cu12 cuml-cu12 cupy-cuda12x --extra-index-url=https://pypi.nvidia.com || {
        echo "Warning: RAPIDS install failed, will use CPU fallback (slower but works)"
    }
fi

# Verify GPU is available
echo ""
echo "[3/4] Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "Warning: nvidia-smi not found, GPU may not be available"
fi

# Verify RAPIDS installation
echo ""
echo "[4/4] Verifying RAPIDS..."
python -c "import cudf; print('  cuDF: OK'); import cuml; print('  cuML: OK')" 2>/dev/null || {
    echo "  Warning: RAPIDS not working, will use CPU fallback"
    echo "  This is slower but will still work"
}

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "  1. Upload your data (run from local machine):"
echo "     scp -P <PORT> -i ~/.ssh/id_ed25519 data/*.csv root@<IP>:/workspace/Hybridyzer/data/"
echo ""
echo "  2. Start training:"
echo "     python train.py --runpod --walkforward"
echo ""
echo "  3. Or run overnight with Thompson sampling:"
echo "     nohup python tools/nightly_runner.py --runpod --time-budget-hours 24 --promote-best --bandit-thompson > training.log 2>&1 &"
echo "     tail -f training.log"
echo ""
