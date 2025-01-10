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
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
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


def setup_dataset():
    logger.info("Setting up dataset...")
    try:
        dataset = CTDataset(
            json_path="../data/dataset.json",
            img_size=384,
            depth=320,
            downsample_ratio=[1.0, 1.0, 1.0],
            cache_dir="../data/cache",
            batch_size=1,
            val_batch_size=1,
            num_workers=4,
            dist=False,
        )
        logger.info("Dataset setup successful")
        return dataset.setup("train"), dataset.train_list, dataset.val_list
    except Exception as e:
        logger.error(f"Failed to setup dataset: {e}")
        raise


def setup_model(device):
    logger.info(f"Setting up model on {device}...")
    try:
        model = VideoMAEForPreTraining.from_pretrained("standardmodelbio/smb-vision-base", trust_remote_code=True).to(
            device
        )
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
        tensors = {"embedding": embedding.last_hidden_state}
        save_file(tensors, save_path)
        logger.info(f"Saved embedding to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save embedding to {save_path}: {e}")
        raise


def process_split(data, file_list, split_name):
    logger.info(f"\nProcessing {split_name} split...")
    error_files = []

    for i, item in enumerate(data[split_name]):
        try:
            image = item["image"]
            logger.info(f"Processing image {i + 1}/{len(data[split_name])} with shape: {image.shape}")

            filepath = Path(file_list[i]["image"])
            save_name = filepath.stem.replace(".nii", "")
            save_path = Path("embeddings") / f"{save_name}.safetensors"

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


if __name__ == "__main__":
    logger.info("Starting embedding generation process")

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        # Setup dataset and model
        data, train_list, val_list = setup_dataset()
        model = setup_model(device)

        # Create output directory
        os.makedirs("embeddings", exist_ok=True)
        logger.info("Created embeddings directory")

        # Process train and validation splits
        process_split(data, train_list, "train")
        process_split(data, val_list, "validation")

        logger.info("Embedding generation process completed successfully")

    except Exception as e:
        logger.error(f"Fatal error in main process: {e}")
        raise
