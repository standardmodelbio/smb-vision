#!/bin/bash

HOME=/workspace/inspect
S3_PATH=s3://smb-data-us-east-2/embeddings/dataset=inspect

# python scripts/inference/run_inspect.py \
#     --image_dir $HOME/CTPA/ \
#     --saved_json_path $HOME/asset-inspect.json \
#     --cache_dir $HOME/cache \
#     --save_dir /workspace/embeddings/dataset=inspect \
#     --bf16

# aws s3 sync /workspace/embeddings/dataset=inspect $S3_PATH --profile smb-dev

python src/encoders/run_merlin.py \
    --encoder merlin \
    --image_dir $HOME/CTPA/ \
    --saved_json_path $HOME/asset-inspect.json \
    --cache_dir $HOME/cache \
    --save_dir /workspace/embeddings/dataset=inspect \
    --bf16

# aws s3 sync /workspace/embeddings/dataset=inspect $S3_PATH --profile smb-dev
