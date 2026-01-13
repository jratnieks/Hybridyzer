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

# Install core dependencies first
echo "[1/4] Installing core dependencies..."
pip install pandas numpy scikit-learn matplotlib pyarrow joblib scipy numba

# Install RAPIDS for GPU acceleration
echo ""
echo "[2/4] Installing RAPIDS GPU libraries..."
pip install cudf-cu12 cuml-cu12 cupy-cuda12x --extra-index-url=https://pypi.nvidia.com || {
    echo "Warning: RAPIDS install failed, will use CPU fallback (slower but works)"
}

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
echo "     nohup python tools/nightly_runner.py --time-budget-hours 24 --promote-best --bandit-thompson > training.log 2>&1 &"
echo "     tail -f training.log"
echo ""
