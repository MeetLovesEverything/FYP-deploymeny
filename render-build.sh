#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install dependencies from requirements.txt (excluding torch)
pip install -r requirements.txt

# Install CPU-only PyTorch explicitly
# This saves massive amounts of space and memory
pip install torch==2.1.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
