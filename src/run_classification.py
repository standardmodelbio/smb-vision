import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch

import transformers
from dataloader.load import SMBVisionDataset
from transformers import (
    AutoConfig,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    VideoMAEConfig,
    VideoMAEForVideoClassification,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.46.0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/image-pretraining/requirements.txt",
)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_path: Optional[str] = field(
        default=None, required=True, metadata={"help": "The local train data path."}
    )
    val_data_path: Optional[str] = field(default=None, required=True, metadata={"help": "The local val data path."})
    task_type: str = field(
        default="classification",
        metadata={"help": "Type of task: 'classification' or 'regression'"},
    )
    num_labels: int = field(
        default=2,
        metadata={"help": "Number of labels for classification task"},
    )
    label_column: str = field(
        default="label",
        metadata={"help": "Name of the label column in the dataset"},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config we are going to fine-tune.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Can be a local path to a pytorch_model.bin or a "
                "checkpoint identifier on the hub."
            )
        },
    )
    config_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store (cache) the pretrained models/datasets downloaded from the hub"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    image_size: Optional[int] = field(
        default=224,
        metadata={"help": "The size (resolution) of each image."},
    )
    depth: Optional[int] = field(
        default=160,
        metadata={"help": "The depth of the 3D volume."},
    )
    patch_size: Optional[int] = field(
        default=16,
        metadata={"help": "The size (resolution) of each patch."},
    )


# Add custom TrainingArguments class
@dataclass
class CustomTrainingArguments(TrainingArguments):
    vision_lr: float = field(
        default=1e-5,
        metadata={"help": "Learning rate for vision model parameters"},
    )
    merger_lr: float = field(
        default=5e-5,
        metadata={"help": "Learning rate for merger parameters"},
    )


def collate_fn(examples):
    # Unpack nested lists (common in MONAI/PyTorch datasets)
    unpacked = []
    for ex in examples:
        while isinstance(ex, (list, tuple)) and len(ex) == 1:
            ex = ex[0]
        unpacked.append(ex)
    examples = unpacked

    # Stack tensors and get labels
    pixel_values = torch.stack([ex["image"] for ex in examples])
    labels = torch.tensor([ex["label"] for ex in examples])

    return {"pixel_values": pixel_values, "labels": labels}


def compute_metrics(eval_pred, data_args):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # For classification
    if data_args.task_type == "classification":
        predictions = predictions.argmax(axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}
    # For regression
    else:
        mse = ((predictions - labels) ** 2).mean()
        return {"mse": mse}


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_classification", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
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

    # Initialize our dataset.
    train_dataset = (
        SMBVisionDataset(data_args.train_data_path, flag="train", cache_dir=model_args.cache_dir)
        if data_args.train_data_path
        else None
    )
    val_dataset = (
        SMBVisionDataset(data_args.val_data_path, flag="val", cache_dir=model_args.cache_dir)
        if data_args.val_data_path
        else None
    )

    # Create config
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name_or_path:
        config = AutoConfig.from_pretrained(model_args.config_name_or_path, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = VideoMAEConfig()
        logger.warning("Training new model from scratch")

    # Update config for classification
    config.update(
        {
            "image_size": model_args.image_size,
            "patch_size": model_args.patch_size,
            "num_channels": 1,
            "num_frames": model_args.depth,
            "tubelet_size": model_args.patch_size,
            "num_labels": data_args.num_labels,
            "problem_type": "single_label_classification" if data_args.task_type == "classification" else "regression",
        }
    )

    # Create model
    if model_args.model_name_or_path:
        model = VideoMAEForVideoClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        logger.info("Training new model from scratch")
        model = VideoMAEForVideoClassification(config)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=val_dataset if training_args.do_eval else None,
        data_collator=collate_fn,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, data_args),
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
        "tasks": data_args.task_type,
        "dataset": data_args.json_path,
        "tags": [data_args.task_type],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
