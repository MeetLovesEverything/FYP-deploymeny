#!/usr/bin/env bash
# Exit on error
set -o errexit

echo "BUILD START: Installing dependencies..."

# Upgrade pip first to avoid issues
pip install --upgrade pip

# Install dependencies from requirements.txt (excluding torch)
pip install -r requirements.txt

echo "BUILD: Installing CPU-only PyTorch..."
# Install CPU-only PyTorch explicitly
pip install torch==2.1.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

echo "BUILD: Verifying installation..."
python -c "import torch; print(f'SUCCESS: PyTorch {torch.__version__} installed')"

echo "BUILD COMPLETE"
