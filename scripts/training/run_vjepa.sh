#!/bin/bash
export WANDB_PROJECT=smb-vision

# python scripts/download_from_s3.py

# download data
# aws s3 sync s3://smb-dev-us-east-2-data/datasets/idc2niix-ct/ /workspace/tcia/ \
#     --exclude "*Eq*" \
#     --exclude "*a.nii.gz*" \
#     --profile smb-dev

# build train file
python scripts/build_train_file.py --directory_path /workspace/tcia/

# train
accelerate launch --config_file acc_configs/multi_gpu_config.yaml src/run_vjepa.py \
    --data_path nifti_files.json \
    --cache_dir /workspace/cache/ \
    --model_name_or_path facebook/vjepa2-vitl-fpc64-256 \
    --lr_scheduler_type cosine_with_min_lr \
    --learning_rate 3e-5 \
    --lr_scheduler_kwargs "{\"min_lr\": 1e-7}" \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --do_train true \
    --do_eval true \
    --output_dir /workspace/saves/vjepa2-vitl-fpc64-256 \
    --overwrite_output_dir true \
    --remove_unused_columns false \
    --eval_strategy "no" \
    --eval_steps 500 \
    --save_steps 100 \
    --save_total_limit 1 \
    --bf16 true \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --report_to wandb \
    --run_name vjepa2-vitl-fpc64-256
