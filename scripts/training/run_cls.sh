#!/bin/bash
export WANDB_PROJECT=smb-vision-cls

# Paths and model configuration
# TODO: change these paths to your own
OUTPUT_DIR=./saves/smb-vision-cls-05202025
DATA_PATH=./dummy_data/dummy_dataset.json
DATA_CACHE_PATH=./cache/
MODEL_NAME=standardmodelbio/smb-vision-base
RUN_NAME=smb-vision-cls-05202025

# Model parameters
# TODO: change these parameters to your own
NUM_LABELS=2
LEARNING_RATE=1e-5
VISION_LR=1e-5
MERGER_LR=5e-5
WEIGHT_DECAY=0.1
MAX_GRAD_NORM=1.0
WARMUP_RATIO=0.01
NUM_EPOCHS=1

# Batch size and device configuration
# TODO: change these parameters to your own
GLOBAL_BATCH_SIZE=256
BATCH_PER_DEVICE=16
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# train
accelerate launch src/run_classification.py \
    --json_path $DATA_PATH \
    --cache_dir $DATA_CACHE_PATH \
    --model_name_or_path $MODEL_NAME \
    --task_type classification \
    --num_labels $NUM_LABELS \
    --lr_scheduler_type cosine \
    --learning_rate $LEARNING_RATE \
    --vision_lr $VISION_LR \
    --merger_lr $MERGER_LR \
    --weight_decay $WEIGHT_DECAY \
    --max_grad_norm $MAX_GRAD_NORM \
    --warmup_ratio $WARMUP_RATIO \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --per_device_eval_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --do_train true \
    --do_eval true \
    --overwrite_output_dir true \
    --remove_unused_columns false \
    --output_dir $OUTPUT_DIR \
    --eval_strategy "steps" \
    --eval_steps 500 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 10 \
    --load_best_model_at_end true \
    --metric_for_best_model "accuracy" \
    --greater_is_better true \
    --bf16 true \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --report_to none \
    --run_name $RUN_NAME
