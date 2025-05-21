import glob
import json
import os
import random
import sys

import fire
from loguru import logger


random.seed(42)


def collect_nifti_files(directory_path, output_file="nifti_files.json", val_size=100):
    # Get all nifti files in directory and subdirectories
    nifti_files = glob.glob(os.path.join(directory_path, "**", "*.nii.gz"), recursive=True)
    logger.info(f"Found {len(nifti_files)} total NIfTI files in {directory_path}")

    # Randomly shuffle files
    random.shuffle(nifti_files)

    # Split into train and val based on validation size
    val_size = min(val_size, len(nifti_files))
    train_files = nifti_files[val_size:]
    val_files = nifti_files[:val_size]

    # Convert to {"image": path} format
    train_records = [{"image": path} for path in train_files]
    val_records = [{"image": path} for path in val_files]

    # Create dictionary
    data_split = {"train": train_records, "validation": val_records}

    # Save to json
    with open(output_file, "w") as f:
        json.dump(data_split, f, indent=2)

    logger.info(
        f"Split dataset into {len(data_split['train'])} training files and {len(data_split['validation'])} validation files"
    )
    logger.info(f"Output saved to {output_file}")

    return data_split


def main(directory_path: str, output_file: str = "nifti_files.json", val_size: int = 100):
    """
    Collect NIfTI files from a directory and split them into training and validation sets.

    Args:
        directory_path: Path to the directory containing NIfTI files
        output_file: Path to save the output JSON file (default: nifti_files.json)
        val_size: Number of files to use for validation (default: 100)
    """
    logger.info(f"Starting to process NIfTI files from {directory_path}")
    return collect_nifti_files(directory_path, output_file, val_size)


if __name__ == "__main__":
    # Configure loguru
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )
    fire.Fire(main)
