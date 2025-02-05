import argparse
import glob
import json
import logging
import os
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import VideoMAEForPreTraining

from dataloader.load import CTDataset


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("processing.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def build_json_from_nifti_files(train_dir, val_dir, output_json_path):
    """Build a json file containing paths to nifti files from separate train/validation directories"""
    # Find all .nii or .nii.gz files in train directory
    train_files = []
    for ext in ["*.nii", "*.nii.gz"]:
        paths = glob.glob(os.path.join(train_dir, "**", ext), recursive=True)
        train_files.extend([{"image": path} for path in paths])

    # Find all .nii or .nii.gz files in validation directory
    val_files = []
    for ext in ["*.nii", "*.nii.gz"]:
        paths = glob.glob(os.path.join(val_dir, "**", ext), recursive=True)
        val_files.extend([{"image": path} for path in paths])

    data_dict = {"train": train_files, "validation": val_files}

    # Write to json file
    with open(output_json_path, "w") as f:
        json.dump(data_dict, f, indent=2)

    logger.info(f"Created dataset JSON file at {output_json_path}")
    return output_json_path


def setup_dataset(args):
    logger.info("Setting up dataset...")
    try:
        dataset = CTDataset(
            json_path=args.json_path,
            img_size=args.img_size,
            depth=args.depth,
            batch_size=args.batch_size,
            val_batch_size=args.val_batch_size,
            num_workers=args.num_workers,
            dist=args.dist,
        )
        logger.info("Dataset setup successful")
        return dataset.setup("train"), dataset.train_list, dataset.val_list
    except Exception as e:
        logger.error(f"Failed to setup dataset: {e}")
        raise


def setup_model(device, model_name):
    logger.info(f"Setting up model on {device}...")
    try:
        model = VideoMAEForPreTraining.from_pretrained(model_name, trust_remote_code=True).to(device)
        logger.info("Model setup successful")
        return model
    except Exception as e:
        logger.error(f"Failed to setup model: {e}")
        raise


def generate_embedding(model, image, device):
    try:
        image = image.to(device)
        with torch.no_grad():
            embedding = model.videomae(image.unsqueeze(0))
        return embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        raise


def save_embedding(embedding, save_path):
    try:
        tensors = {"embedding": embedding.last_hidden_state.mean(dim=1)}
        save_file(tensors, save_path)
        logger.info(f"Saved embedding to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save embedding to {save_path}: {e}")
        raise


def process_split(data, file_list, split_name, model, device, output_dir):
    logger.info(f"\nProcessing {split_name} split...")
    error_files = []

    for i, item in enumerate(data[split_name]):
        try:
            image = item["image"]
            logger.info(f"Processing image {i + 1}/{len(data[split_name])} with shape: {image.shape}")

            filepath = Path(file_list[i]["image"])
            save_name = filepath.stem.replace(".nii", "")
            save_path = Path(output_dir) / f"{save_name}.safetensors"

            embedding = generate_embedding(model, image, device)
            save_embedding(embedding, save_path)

        except Exception as e:
            error_msg = f"Error processing {filepath}: {str(e)}"
            logger.error(error_msg)
            error_files.append({"file": str(filepath), "error": str(e)})

    if error_files:
        logger.error(f"Failed to process {len(error_files)} files in {split_name} split")
        with open(f"error_files_{split_name}.json", "w") as f:
            json.dump(error_files, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate embeddings from medical images")
    parser.add_argument("--json_path", type=str, default="../data/dataset.json", help="Path to dataset JSON file")
    parser.add_argument("--img_size", type=int, default=512, help="Image size")
    parser.add_argument("--depth", type=int, default=320, help="Image depth")
    parser.add_argument("--cache_dir", type=str, default="../data/cache", help="Cache directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=1, help="Validation batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--dist", action="store_true", help="Enable distributed training")
    parser.add_argument(
        "--model_name", type=str, default="standardmodelbio/smb-vision-base-20250122", help="Model name or path"
    )
    parser.add_argument("--output_dir", type=str, default="embeddings", help="Output directory for embeddings")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    return parser.parse_args()


if __name__ == "__main__":
    logger.info("Starting embedding generation process")
    args = parse_args()

    # Setup device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        # Setup dataset and model
        data, train_list, val_list = setup_dataset(args)
        model = setup_model(device, args.model_name)

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Created embeddings directory at {args.output_dir}")

        # Process train and validation splits
        process_split(data, train_list, "train", model, device, args.output_dir)
        process_split(data, val_list, "validation", model, device, args.output_dir)

        logger.info("Embedding generation process completed successfully")

    except Exception as e:
        logger.error(f"Fatal error in main process: {e}")
        raise
