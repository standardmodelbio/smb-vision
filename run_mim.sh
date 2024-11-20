#!/bin/bash
export WANDB_PROJECT=smb-vision
export WANDB_LOG_MODEL=checkpoint

torchrun --nproc_per_node 8 src/run_mim.py \
    --json_path ./smb-vision-large-train-mim.json \
    --cache_dir ../cache/ \
    --learning_rate 3e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --warmup_steps 500 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --do_train true \
    --do_eval true \
    --overwrite_output_dir true \
    --output_dir ./saves/smb-vision-large-1120 \
    --eval_strategy "no" \
    --eval_steps 500 \
    --save_steps 500 \
    --gradient_checkpointing true \
    --bf16 true \
    --logging_steps 1 \
    --report_to wandb \
    --run_name smb-vision-large-1120
