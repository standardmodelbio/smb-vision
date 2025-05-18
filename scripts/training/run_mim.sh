#!/bin/bash
export WANDB_PROJECT=smb-vision
# export WANDB_LOG_MODEL=checkpoint

# python scripts/download_from_s3.py

# download data
# aws s3 sync s3://smb-dev-us-east-2-data/datasets/idc2niix-ct/ ../nifti_files/

# build train file
# python scripts/build_train_file.py

# train
accelerate launch src/run_mim.py \
    --json_path /workspace/smb_vision_dataset.json \
    --cache_dir /workspace/cache/ \
    --model_name_or_path standardmodelbio/smb-vision-base \
    --lr_scheduler_type cosine \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.01 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --do_train true \
    --do_eval true \
    --overwrite_output_dir true \
    --remove_unused_columns false \
    --output_dir /workspace/saves/smb-vision-base-05152025 \
    --eval_strategy "no" \
    --eval_steps 500 \
    --save_steps 5000 \
    --bf16 true \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --report_to wandb \
    --run_name smb-vision-base-05152025
