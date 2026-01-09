#!/bin/bash
# RunPod setup script for Hybridyzer
# Run this after cloning to a network volume

set -e

echo "=== Hybridyzer RunPod Setup ==="

# Check if we're in the right directory
if [ ! -f "train.py" ]; then
    echo "Error: Run this script from the Hybridyzer root directory"
    exit 1
fi

# Install micromamba if not available
if ! command -v micromamba &> /dev/null && ! command -v conda &> /dev/null; then
    echo "Installing micromamba..."
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
    export PATH="$PWD/bin:$PATH"
    echo 'export PATH="$PWD/bin:$PATH"' >> ~/.bashrc
fi

# Use whichever is available
if command -v micromamba &> /dev/null; then
    CONDA_CMD="micromamba"
elif command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
else
    CONDA_CMD="conda"
fi

echo "Using $CONDA_CMD for environment management"

# Create/update the environment
echo "Creating hybridyzer environment..."
$CONDA_CMD env create -f environment.runpod.yml -y || $CONDA_CMD env update -f environment.runpod.yml -y

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Activate the environment:"
echo "       $CONDA_CMD activate hybridyzer"
echo ""
echo "  2. Copy your data files to data/ directory:"
echo "       - btcusd_5min_train_2017_2022.csv"
echo "       - btcusd_5min_val_2023.csv"
echo "       - btcusd_5min_test_2024.csv"
echo ""
echo "  3. Run training:"
echo "       python train.py --runpod --walkforward"
echo ""
echo "  4. Or run the nightly runner:"
echo "       python tools/nightly_runner.py --time-budget-hours 8 --promote-best"
echo ""
