import copy
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from dataloader.load import CTPersistentDataset
from dataloader.transforms import ct_transforms
from models.vjepa.modeling_vjepa import VJEPA2Config, VJEPA2Model
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    get_last_checkpoint,
    send_example_telemetry,
)


random.seed(42)

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""

    data_path: Optional[str] = field(default=None, metadata={"help": "The local data path."})
    mask_patch_size: int = field(
        default=16,
        metadata={"help": "The size of the square patches to use for masking."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this value if set."
        },
    )
    attn_implementation: str = field(
        default="flash_attention_2",
        metadata={"help": "Attention implementation to use"},
    )


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config we are going to pre-train."""

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
        },
    )
    config_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models/datasets downloaded from the hub"},
    )
    image_size: Optional[int] = field(
        default=384,
        metadata={"help": "The size (resolution) of each image."},
    )
    depth: Optional[int] = field(
        default=256,
        metadata={"help": "The depth of the 3D volume."},
    )
    patch_size: Optional[int] = field(
        default=16,
        metadata={"help": "The size (resolution) of each patch."},
    )


class MomentumEncoder:
    """Momentum-based EMA update for target encoder"""

    def __init__(self, model, momentum=0.99925):
        self.model = model
        self.momentum = momentum

    def update(self, source_model):
        """Update target encoder weights using momentum"""
        with torch.no_grad():
            for param_q, param_k in zip(source_model.parameters(), self.model.parameters()):
                param_k.data.mul_(self.momentum).add_(param_q.data, alpha=1.0 - self.momentum)


class VJEPATrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_encoder = copy.deepcopy(self.model)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        self.momentum_encoder = MomentumEncoder(self.target_encoder, momentum=0.99925)
        self.loss_fn = nn.L1Loss()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the V-JEPA loss using L1 loss between predicted target and momentum encoded target.
        """
        pixel_values = inputs["pixel_values"]
        context_mask = inputs["context_mask"]
        target_mask = inputs["target_mask"]

        # Forward pass through main model
        outputs = model(
            pixel_values_videos=pixel_values,
            context_mask=[context_mask],
            target_mask=[target_mask],
            skip_predictor=False,
        )

        # Get predicted target from predictor
        predicted_target = outputs.predictor_output.last_hidden_state

        # Get target from momentum encoder
        with torch.no_grad():
            target_outputs = self.target_encoder(
                pixel_values_videos=pixel_values,
                context_mask=[context_mask],
                target_mask=[target_mask],
                skip_predictor=True,
            )
            target = target_outputs.target_hidden_state

        # Compute L1 loss
        loss = self.loss_fn(predicted_target, target)

        # Update momentum encoder
        self.momentum_encoder.update(model)

        return (loss, outputs) if return_outputs else loss


def collate_fn(examples):
    # Unpack nested lists (common in MONAI/PyTorch datasets)
    unpacked = []
    for ex in examples:
        # Unpack until we get to the dictionary
        while isinstance(ex, (list, tuple)) and len(ex) == 1:
            ex = ex[0]
        unpacked.append(ex)
    examples = unpacked

    # Stack tensors and rename keys
    pixel_values = torch.stack([ex["image"] for ex in examples])
    random_example = random.choice(examples)
    context_mask = torch.stack([random_example["context_mask"] for _ in examples])
    target_mask = torch.stack([random_example["target_mask"] for _ in examples])

    return {"pixel_values": pixel_values, "context_mask": [context_mask], "target_mask": [target_mask]}


def main():
    # See all possible arguments in src/transformers/training_args.py
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them.
    send_example_telemetry("run_vjepa", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Create dataset
    train_dataset = CTPersistentDataset(
        data_path=data_args.data_path,
        split="train",
        transform=ct_transforms["vjepa"],
        cache_dir=model_args.cache_dir,
    )
    eval_dataset = CTPersistentDataset(
        data_path=data_args.data_path,
        split="val",
        transform=ct_transforms["vjepa"],
        cache_dir=model_args.cache_dir,
    )

    # Create config
    config = VJEPA2Config.from_pretrained(model_args.model_name_or_path)
    config.update(
        {
            "image_size": model_args.image_size,
            "crop_size": model_args.image_size,
            "patch_size": model_args.patch_size,
            "in_chans": 1,
            "frames_per_clip": model_args.depth,
            "tubelet_size": model_args.patch_size,
            "torch_dtype": torch.bfloat16,
            "attn_implementation": training_args.attn_implementation,
        }
    )

    # Initialize model
    # if model_args.model_name_or_path:
    #     model = VJEPA2Model.from_pretrained(
    #         model_args.model_name_or_path,
    #         config=config,
    #         cache_dir=model_args.cache_dir,
    #         torch_dtype=torch.bfloat16,
    #         attn_implementation=training_args.attn_implementation,
    #     )
    # else:
    logger.info("Training new model from scratch")
    model = VJEPA2Model(config)

    # Initialize trainer
    trainer = VJEPATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        compute_metrics=lambda eval_pred: {"loss": eval_pred.predictions[0].item()},
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Write model card and (optionally) push to hub
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "video-jepa",
        "tags": ["video-jepa"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
