import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
# from monai.data.utils import pad_list_data_collate

import transformers
from dataloader.mim import MIMDataset
from transformers import (
    MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING,
    AutoConfig,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    VideoMAEConfig,
    VideoMAEForPreTraining,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


""" Pre-training a 🤗 Transformers model for simple masked image modeling (SimMIM).
Any model supported by the AutoModelForMaskedImageModeling API can be used.
"""

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.46.0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/image-pretraining/requirements.txt",
)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to
    specify them on the command line.
    """

    dataset_name: Optional[str] = field(
        default="cifar10",
        metadata={"help": "Name of a dataset from the datasets package"},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."},
    )
    json_path: Optional[str] = field(default=None, metadata={"help": "The local json data path."})
    image_column_name: Optional[str] = field(
        default="image",
        metadata={"help": "The column name of the images in the files. If not set, will try to use 'image' or 'img'."},
    )
    train_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the training data."})
    validation_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the validation data."})
    train_val_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )
    mask_patch_size: int = field(
        default=16,
        metadata={"help": "The size of the square patches to use for masking."},
    )
    mask_ratio: float = field(
        default=0.5,
        metadata={"help": "Percentage of patches to mask."},
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

    def __post_init__(self):
        data_files = {}
        if self.train_dir is not None:
            data_files["train"] = self.train_dir
        if self.validation_dir is not None:
            data_files["val"] = self.validation_dir
        self.data_files = data_files if data_files else None


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/image processor we are going to pre-train.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Can be a local path to a pytorch_model.bin or a "
                "checkpoint identifier on the hub. "
                "Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store (cache) the pretrained models/datasets downloaded from the hub"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
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
        metadata={
            "help": (
                "The size (resolution) of each image. If not specified, will use `image_size` of the configuration."
            )
        },
    )
    depth: Optional[int] = field(
        default=160,
        metadata={"help": ("The depth of the 3D volume.")},
    )
    patch_size: Optional[int] = field(
        default=16,
        metadata={
            "help": (
                "The size (resolution) of each patch. If not specified, will use `patch_size` of the configuration."
            )
        },
    )
    encoder_stride: Optional[int] = field(
        default=None,
        metadata={"help": "Stride to use for the encoder."},
    )


def collate_fn(examples):
    # Unpack nested lists (common in MONAI/PyTorch datasets)
    unpacked = []
    for ex in examples:
        # Unpack until we get to the dictionary
        while isinstance(ex, (list, tuple)) and len(ex) == 1:
            ex = ex[0]
        unpacked.append(ex)
    examples = unpacked

    # Debug: Print first example's structure
    # print("\nFirst unpacked example keys:", examples[0].keys() if isinstance(examples[0], dict) else "Not a dict")

    # Verify all examples have required keys
    # if not all(isinstance(ex, dict) and "image" in ex and "mask" in ex for ex in examples):
    #     bad_indices = [
    #         i for i, ex in enumerate(examples) if not (isinstance(ex, dict) and "image" in ex and "mask" in ex)
    #     ]
    #     print(f"ERROR: Missing keys in examples at indices: {bad_indices}")
    #     raise ValueError("Batch contains examples without 'image' or 'mask' keys")

    # Stack tensors and rename keys
    pixel_values = torch.stack([ex["image"] for ex in examples])
    masks = torch.stack([ex["mask"] for ex in examples])

    return {"pixel_values": pixel_values, "bool_masked_pos": masks}


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_mim", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
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
    dataset = MIMDataset(
        json_path=data_args.json_path,
        img_size=model_args.image_size,
        depth=model_args.depth,
        mask_patch_size=data_args.mask_patch_size,
        patch_size=model_args.patch_size,
        downsample_ratio=(1.5, 1.5, 3.0),
        cache_dir=model_args.cache_dir,
        dist=False,
        mask_ratio=data_args.mask_ratio,
    )
    datasets = dataset.setup("train")

    # Create config
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
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

    # make sure the decoder_type is "simmim" (only relevant for BEiT)
    # if hasattr(config, "decoder_type"):
    #     config.decoder_type = "simmim"

    # adapt config
    model_args.image_size = model_args.image_size if model_args.image_size is not None else config.image_size
    model_args.patch_size = model_args.patch_size if model_args.patch_size is not None else config.patch_size
    model_args.depth = model_args.depth if model_args.depth is not None else config.num_frames
    # model_args.encoder_stride = (
    #     model_args.encoder_stride if model_args.encoder_stride is not None else config.encoder_stride
    # )

    config.update(
        {
            "image_size": model_args.image_size,
            "patch_size": model_args.patch_size,
            "num_channels": 1,
            "num_frames": model_args.depth,
            "tubelet_size": model_args.patch_size,
        }
    )

    # create image processor
    # if model_args.image_processor_name:
    #     image_processor = AutoImageProcessor.from_pretrained(
    #         model_args.image_processor_name, **config_kwargs
    #     )
    # elif model_args.model_name_or_path:
    #     image_processor = AutoImageProcessor.from_pretrained(
    #         model_args.model_name_or_path, **config_kwargs
    #     )
    # else:
    #     image_processor = VideoMAEImageProcessor()

    # create model
    if model_args.model_name_or_path:
        model = VideoMAEForPreTraining.from_pretrained(
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
        model = VideoMAEForPreTraining(config)

    # if training_args.do_train:
    #     column_names = ds["train"].column_names
    # else:
    #     column_names = ds["validation"].column_names

    # if data_args.image_column_name is not None:
    #     image_column_name = data_args.image_column_name
    # elif "image" in column_names:
    #     image_column_name = "image"
    # elif "img" in column_names:
    #     image_column_name = "img"
    # else:
    #     image_column_name = column_names[0]

    # transformations as done in original SimMIM paper
    # source: https://github.com/microsoft/SimMIM/blob/main/data/data_simmim.py
    # transforms = Compose(
    #     [
    #         Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
    #         RandomResizedCrop(model_args.image_size, scale=(0.67, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
    #         RandomHorizontalFlip(),
    #         ToTensor(),
    #         Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
    #     ]
    # )

    # create mask generator
    # mask_generator = MaskGenerator(
    #     input_size=model_args.image_size,
    #     mask_patch_size=data_args.mask_patch_size,
    #     model_patch_size=model_args.patch_size,
    #     mask_ratio=data_args.mask_ratio,
    # )

    # def preprocess_images(examples):
    #     """Preprocess a batch of images by applying transforms + creating a corresponding mask, indicating
    #     which patches to mask."""

    #     examples["pixel_values"] = [transforms(image) for image in examples[image_column_name]]
    #     examples["mask"] = [mask_generator() for i in range(len(examples[image_column_name]))]

    #     return examples

    # if training_args.do_train:
    #     if "train" not in ds:
    #         raise ValueError("--do_train requires a train dataset")
    #     if data_args.max_train_samples is not None:
    #         ds["train"] = ds["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
    #     # Set the training transforms
    #     ds["train"].set_transform(preprocess_images)

    # if training_args.do_eval:
    #     if "validation" not in ds:
    #         raise ValueError("--do_eval requires a validation dataset")
    #     if data_args.max_eval_samples is not None:
    #         ds["validation"] = (
    #             ds["validation"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
    #         )
    #     # Set the validation transforms
    #     ds["validation"].set_transform(preprocess_images)

    # Initialize our trainer
    print("\nInitializing trainer...")
    # print("Training dataset type:", type(ds_train))
    print("First training item type:", type(datasets["train"][0]))
    print(
        "First training item keys:",
        datasets["train"][0].keys() if isinstance(datasets["train"][0], dict) else "Not a dict",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"] if training_args.do_train else None,
        eval_dataset=datasets["validation"] if training_args.do_eval else None,
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
        "tasks": "masked-image-modeling",
        "dataset": data_args.dataset_name,
        "tags": ["masked-image-modeling"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
