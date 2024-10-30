#!/bin/bash
export WANDB_PROJECT=smb-vision
export WANDB_LOG_MODEL=checkpoint

python run_mim.py \
    --json_path ../data/lung-ct-4k-mim.json \
    --cache_dir ../cache/ \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --do_train true \
    --do_eval true \
    --overwrite_output_dir true \
    --output_dir ./saves/dry_run/smb-vision-base-1029 \
    --eval_strategy "steps" \
    --eval_steps 5 \
    --save_steps 1000 \
    --gradient_checkpointing true \
    --bf16 true \
    --logging_steps 1 \
    --report_to wandb \
    --run_name smb-vision-base-1029
