#!/bin/bash

HOME=/workspace/inspect

python scripts/inference/run_inspect.py \
    --image_dir $HOME/CTPA/ \
    --saved_json_path $HOME/asset-inspect.json \
    --cache_dir $HOME/cache \
    --save_path /workspace/embeddings/dataset=inspect \
    --bf16
