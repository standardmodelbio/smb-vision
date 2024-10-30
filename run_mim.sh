#!/bin/bash
python run_mim.py \
    --json_path ./train.json \
    --cache_dir ./cache/ \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --do_train true \
    --do_eval true \
    --overwrite_output_dir true \
    --output_dir ./saves/dry_run/smb-vision-base-1027 \
    --eval_strategy "steps" \
    --eval_steps 100 \
    --save_steps 1000 \
    --gradient_checkpointing true \
    --bf16 true \
    --logging_steps 1
