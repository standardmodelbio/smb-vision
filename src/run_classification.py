import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import evaluate
import numpy as np
import torch

import transformers
from dataloader.load import CTPersistentDataset
from dataloader.transforms import ct_transforms
from models.dinov2.modeling_dinov2 import Dinov2ForImageClassification
from models.videomae.modeling_videomae import VideoMAEForVideoClassification
from transformers import (
    AutoConfig,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    VideoMAEConfig,
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


def cox_ph_loss_sorted(log_h: torch.Tensor, events: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    r"""Requires the input to be sorted by descending duration time.

    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1)
    log_h = log_h.view(-1)
    gamma = log_h.max()
    log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
    return -log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum() + eps)


def cox_loss(
    risk_scores: torch.Tensor, durations: torch.Tensor, events: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    r"""Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.

    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    # Sort by duration in descending order
    idx = durations.sort(descending=True)[1]
    events = events[idx]
    log_h = risk_scores[idx]
    return cox_ph_loss_sorted(log_h, events, eps)


class SurvivalTrainer(Trainer):
    def __init__(self, *args, data_args=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_args = data_args

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Handle non-survival tasks with default loss
        if self.data_args.task_type not in ["survival", "cox_regression"]:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        # Forward pass to get risk scores
        outputs = model(inputs["pixel_values"])
        risk_scores = outputs.logits.squeeze(-1)  # (batch_size, 1) -> (batch_size,)

        # Extract labels
        labels = inputs["labels"]
        durations = labels["duration"]
        events = labels["event"]

        # Compute Cox loss
        loss = cox_loss(risk_scores, durations, events)

        return (loss, outputs) if return_outputs else loss


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_path: Optional[str] = field(default=None, metadata={"help": "The local train data path."})
    val_data_path: Optional[str] = field(default=None, metadata={"help": "The local val data path."})
    task_type: str = field(
        default="classification",
        metadata={
            "help": "Type of task: 'classification', 'multilabel_classification', 'regression', 'survival', or 'cox_regression'"
        },
    )
    num_labels: int = field(
        default=2,
        metadata={"help": "Number of labels for classification task"},
    )
    label_columns: List[str] = field(
        default_factory=lambda: ["label"],
        metadata={
            "help": "List of label column names in the dataset for multilabel classification or survival analysis"
        },
    )
    additional_feature_columns: Optional[List[str]] = field(
        default_factory=lambda: [],
        metadata={
            "help": "List of additional feature column names to use alongside the image data for classification. Leave empty for no additional features."
        },
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
    attn_implementation: str = field(
        default="flash_attention_2",
        metadata={"help": "Attention implementation to use"},
    )


def collate_fn(examples, data_args):
    # Unpack nested lists (common in MONAI/PyTorch datasets)
    unpacked = []
    for ex in examples:
        while isinstance(ex, (list, tuple)) and len(ex) == 1:
            ex = ex[0]
        unpacked.append(ex)
    examples = unpacked

    # Stack tensors and get labels
    pixel_values = torch.stack([ex["image"] for ex in examples])

    # Handle additional features if specified
    additional_features = None
    if data_args.additional_feature_columns:
        additional_features = torch.stack(
            [
                torch.tensor([ex[col] for col in data_args.additional_feature_columns], dtype=torch.float32)
                for ex in examples
            ]
        )

    # Handle different task types
    if data_args.task_type == "multilabel_classification":
        # Stack all labels into a single tensor of shape (batch_size, num_labels)
        labels = torch.stack(
            [
                torch.tensor([ex[label_col] for label_col in data_args.label_columns], dtype=torch.float32)
                for ex in examples
            ]
        )
    elif data_args.task_type in ["survival", "cox_regression"]:
        # For survival analysis, we need both duration and event
        labels = {
            "duration": torch.tensor([ex["os"] for ex in examples], dtype=torch.float32),
            "event": torch.tensor([ex["os_event"] for ex in examples], dtype=torch.float32),
        }
    else:
        label_col = data_args.label_columns[0]
        labels = torch.tensor([ex[label_col] for ex in examples])

    result = {"pixel_values": pixel_values, "labels": labels}
    if additional_features is not None:
        result["additional_features"] = additional_features
    return result


def compute_metrics(eval_pred, data_args):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Survival analysis metrics
    if data_args.task_type in ["survival", "cox_regression"]:
        risk_scores = predictions.squeeze()
        true_duration = labels["duration"]
        true_event = labels["event"]

        # Flatten arrays if necessary
        if risk_scores.ndim > 1:
            risk_scores = risk_scores.squeeze(1)
        if true_duration.ndim > 1:
            true_duration = true_duration.squeeze(1)
        if true_event.ndim > 1:
            true_event = true_event.squeeze(1)

        # Calculate C-index
        from lifelines.utils import concordance_index

        c_index = concordance_index(true_duration, risk_scores, true_event)
        return {"c_index": c_index}

    # For multilabel classification
    elif data_args.task_type == "multilabel_classification":
        # Convert logits to multi-hot encoding
        preds = np.array([np.where(p > 0, 1, 0) for p in predictions])

        metric = evaluate.load("f1", config_name="multilabel", cache_dir=None)
        result = metric.compute(predictions=preds, references=labels, average="micro")

        # Add additional metrics
        from sklearn.metrics import precision_score, recall_score

        precision = precision_score(labels, preds, average="micro", zero_division=0)
        recall = recall_score(labels, preds, average="micro", zero_division=0)

        result.update(
            {
                "precision": precision,
                "recall": recall,
            }
        )

        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()

        return result

    # For single-label classification
    elif data_args.task_type == "classification":
        # Convert logits to class predictions
        preds = np.argmax(predictions, axis=1)

        # Load metrics
        accuracy_metric = evaluate.load("accuracy", cache_dir=None)
        auc_metric = evaluate.load("roc_auc", cache_dir=None)

        # Compute base metrics
        result = accuracy_metric.compute(predictions=preds, references=labels)

        # For ROC AUC, we need probability scores for each class
        if predictions.ndim > 1:  # If we have probability scores
            auc_result = auc_metric.compute(prediction_scores=predictions[:, 1], references=labels)
            result.update(auc_result)
        else:
            auc_result = auc_metric.compute(prediction_scores=predictions, references=labels)
            result.update(auc_result)

        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()

        return result

    # For regression
    else:
        preds = np.squeeze(predictions)

        metric = evaluate.load("mse", cache_dir=None)
        result = metric.compute(predictions=preds, references=labels)

        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()

        return result


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
        # Handle empty additional_feature_columns
        if data_args.additional_feature_columns == [""]:
            data_args.additional_feature_columns = None

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
        CTPersistentDataset(
            data_args.train_data_path,
            split="train",
            transform=ct_transforms["smb-vision"],
            cache_dir=model_args.cache_dir,
        )
        if data_args.train_data_path
        else None
    )
    val_dataset = (
        CTPersistentDataset(
            data_args.val_data_path, split="val", transform=ct_transforms["smb-vision"], cache_dir=model_args.cache_dir
        )
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

    # Update config for classification or survival
    config.update(
        {
            "image_size": model_args.image_size,
            "patch_size": model_args.patch_size,
            "num_channels": 1,
            "num_frames": model_args.depth,
            "tubelet_size": model_args.patch_size,
            "num_labels": 1
            if data_args.task_type in ["survival", "cox_regression"]
            else (
                len(data_args.label_columns)
                if data_args.task_type == "multilabel_classification"
                else data_args.num_labels
            ),
            "problem_type": (
                "regression"
                if data_args.task_type in ["survival", "cox_regression"]
                else "multi_label_classification"
                if data_args.task_type == "multilabel_classification"
                else "single_label_classification"
            ),
        }
    )

    # Add additional features size to config if specified
    if data_args.additional_feature_columns:
        config.update({"additional_features_size": len(data_args.additional_feature_columns)})

    # Create model
    if "dino" in model_args.model_name_or_path:
        logger.info("Loading pretrained Dinov2 model")
        model = Dinov2ForImageClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            attn_implementation=training_args.attn_implementation,
        )
    else:
        logger.info("Loading pretrained VideoMAE model")
        model = VideoMAEForVideoClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            attn_implementation=training_args.attn_implementation,
        )

    # Initialize trainer with custom trainer for survival tasks
    trainer_class = SurvivalTrainer if data_args.task_type in ["survival", "cox_regression"] else Trainer
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset if training_args.do_train else None,
        "eval_dataset": val_dataset if training_args.do_eval else None,
        "data_collator": lambda examples: collate_fn(examples, data_args),
        "compute_metrics": lambda eval_pred: compute_metrics(eval_pred, data_args),
    }

    # Only add data_args for SurvivalTrainer
    if trainer_class == SurvivalTrainer:
        trainer_kwargs["data_args"] = data_args

    trainer = trainer_class(**trainer_kwargs)

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
        "dataset": data_args.train_data_path,
        "tags": [data_args.task_type],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
