#!/bin/bash
export WANDB_PROJECT=smb-vision
# export WANDB_LOG_MODEL=checkpoint

# python scripts/download_from_s3.py

# download data
# aws s3 sync s3://smb-dev-us-east-2-data/datasets/idc2niix-ct/ ../nifti_files/

# build train file
# python scripts/build_train_file.py

# train
torchrun --nproc_per_node=1 src/run_mim.py \
    --json_path /home/user/smb_vision_dataset.json \
    --cache_dir /home/user/cache/ \
    --model_name_or_path standardmodelbio/smb-vision-base \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.01 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --do_train true \
    --do_eval true \
    --overwrite_output_dir true \
    --output_dir /home/user/saves/smb-vision-base-05152025 \
    --eval_strategy "no" \
    --eval_steps 500 \
    --save_steps 500 \
    --bf16 true \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --report_to wandb \
    --run_name smb-vision-base-05152025
