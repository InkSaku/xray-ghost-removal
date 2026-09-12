#!/bin/bash
# Setup script for remote training machine
# Run: bash scripts/setup_remote.sh

set -e

echo "=== Setting up remote environment ==="

# Create conda env if needed
if ! conda env list | grep -q "xray"; then
    echo "Creating conda environment..."
    conda create -n xray python=3.12 -y
fi

echo "Activating environment..."
source activate xray 2>/dev/null || conda activate xray

echo "Installing dependencies..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install pydicom scipy matplotlib numpy

echo "Creating directories..."
mkdir -p data/raw/残影图像
mkdir -p data/processed/unet_cleaned
mkdir -p results/figures/unet

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Upload DICOM files to data/raw/残影图像/"
echo "  2. Train:  python scripts/train_unet.py --epochs 100 --batch-size 16"
echo "  3. Or quick test: python scripts/train_unet.py --epochs 10 --samples-per-epoch 500"
echo ""
