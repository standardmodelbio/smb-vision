import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from loguru import logger


def create_dummy_nifti(
    output_path: str,
    shape: tuple = (224, 224, 160),
    num_files: int = 10,
    num_classes: int = 2,
    train_ratio: float = 0.8,
):
    """
    Create dummy NIfTI files and a JSON file with labels for testing.

    Args:
        output_path: Directory to save the dummy data
        shape: Shape of the NIfTI volumes (height, width, depth)
        num_files: Number of NIfTI files to create
        num_classes: Number of classes for classification
        train_ratio: Ratio of training samples
    """
    # Create output directory
    output_path = Path(output_path)
    nifti_dir = output_path / "nifti_files"
    nifti_dir.mkdir(parents=True, exist_ok=True)

    # Generate random data
    train_files = []
    val_files = []

    # Calculate split
    num_train = int(num_files * train_ratio)
    num_val = num_files - num_train

    logger.info(f"Creating {num_train} training and {num_val} validation samples")

    # Create training files
    for i in range(num_train):
        # Generate random 3D volume
        data = np.random.randn(*shape).astype(np.float32)
        # Add some structure to make it look more like a real scan
        data = data + np.sin(np.linspace(0, 10, shape[0]))[:, None, None]
        data = data + np.cos(np.linspace(0, 10, shape[1]))[None, :, None]
        data = data + np.sin(np.linspace(0, 10, shape[2]))[None, None, :]

        # Create NIfTI file
        nifti_file = nifti_dir / f"train_sample_{i:03d}.nii.gz"
        nifti_img = nib.Nifti1Image(data, np.eye(4))
        nifti_img.to_filename(nifti_file)

        # Add to training list with random label
        train_files.append({"image": str(nifti_file), "label": np.random.randint(0, num_classes)})

    # Create validation files
    for i in range(num_val):
        # Generate random 3D volume
        data = np.random.randn(*shape).astype(np.float32)
        # Add some structure to make it look more like a real scan
        data = data + np.sin(np.linspace(0, 10, shape[0]))[:, None, None]
        data = data + np.cos(np.linspace(0, 10, shape[1]))[None, :, None]
        data = data + np.sin(np.linspace(0, 10, shape[2]))[None, None, :]

        # Create NIfTI file
        nifti_file = nifti_dir / f"val_sample_{i:03d}.nii.gz"
        nifti_img = nib.Nifti1Image(data, np.eye(4))
        nifti_img.to_filename(nifti_file)

        # Add to validation list with random label
        val_files.append({"image": str(nifti_file), "label": np.random.randint(0, num_classes)})

    # Create JSON file
    json_data = {"train": train_files, "validation": val_files}

    json_file = output_path / "dummy_dataset.json"
    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=2)

    logger.info(f"Created {num_files} dummy NIfTI files in {nifti_dir}")
    logger.info(f"Created dataset JSON file at {json_file}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Training samples: {len(train_files)}")
    logger.info(f"Validation samples: {len(val_files)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create dummy data for testing")
    parser.add_argument("--output_path", type=str, default="./dummy_data", help="Directory to save the dummy data")
    parser.add_argument(
        "--shape", type=int, nargs=3, default=[224, 224, 160], help="Shape of the NIfTI volumes (height width depth)"
    )
    parser.add_argument("--num_files", type=int, default=10, help="Number of NIfTI files to create")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes for classification")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training samples")

    args = parser.parse_args()

    # Configure logger
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )

    create_dummy_nifti(
        output_path=args.output_path,
        shape=tuple(args.shape),
        num_files=args.num_files,
        num_classes=args.num_classes,
        train_ratio=args.train_ratio,
    )
