#!/bin/bash
export WANDB_PROJECT=smb-vision-cls

# Paths and model configuration
# TODO: change these paths to your own
DATA_PATH=/workspace/data/mdanderson_dataset.parquet
DATA_CACHE_PATH=/workspace/cache/
MODEL_NAME=facebook/dinov2-base
OUTPUT_DIR=/workspace/saves/dinov2-base-survival-mdacc
RUN_NAME=dinov2-base-survival-mdacc

# Model parameters
# TODO: change these parameters to your own
NUM_LABELS=2
LEARNING_RATE=3e-4
VISION_LR=3e-4
MERGER_LR=3e-4
WEIGHT_DECAY=1e-5
WARMUP_RATIO=0.01
NUM_EPOCHS=3

# Batch size and device configuration
# TODO: change these parameters to your own
GLOBAL_BATCH_SIZE=24
BATCH_PER_DEVICE=24
NUM_DEVICES=1
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# train
accelerate launch src/run_classification.py \
    --train_data_path $DATA_PATH \
    --val_data_path $DATA_PATH \
    --cache_dir $DATA_CACHE_PATH \
    --model_name_or_path $MODEL_NAME \
    --task_type classification \
    --num_labels $NUM_LABELS \
    --label_columns "one_year_survival" \
    --lr_scheduler_type cosine \
    --learning_rate $LEARNING_RATE \
    --vision_lr $VISION_LR \
    --merger_lr $MERGER_LR \
    --weight_decay $WEIGHT_DECAY \
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
    --eval_steps 2 \
    --save_strategy "steps" \
    --save_steps 2 \
    --save_total_limit 1 \
    --load_best_model_at_end true \
    --metric_for_best_model "accuracy" \
    --greater_is_better true \
    --bf16 true \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --report_to wandb \
    --run_name $RUN_NAME
