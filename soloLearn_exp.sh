#!/bin/bash
set -euo pipefail

cd learning/solo-learn
pip3 install .[dali,umap,h5] --extra-index-url https://developer.download.nvidia.com/compute/redist
cd ../..
python3 scripts/run.py --repo_dir ./learning/solo-learn \
  -- \
  --config-path scripts/pretrain/cifar/ \
  --config-name simclr.yaml \
  ++name="simclr-cifar100" \
  ++data.dataset=cifar100 
#   wandb.enabled=False