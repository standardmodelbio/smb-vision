#!/bin/bash

HOME=/workspace/inspect

python scripts/inference/run_inspect.py \
    --impressions_path $HOME/Final_Impressions.csv \
    --image_dir $HOME/CTPA/ \
    --saved_json_path $HOME/asset-inspect.json \
    --cache_dir $HOME/cache \
    --save_path s3://smb-data-us-east-2/embeddings/inspect \
    --bf16
