import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from loguru import logger


def create_dummy_nifti(
    output_path: str,
    shape: tuple = (224, 224, 160),
    num_files: int = 10,
    num_classes: int = 2,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
):
    """
    Create dummy NIfTI files and a parquet file with multiple histology labels for testing.

    Args:
        output_path: Directory to save the dummy data
        shape: Shape of the NIfTI volumes (height, width, depth)
        num_files: Number of NIfTI files to create
        num_classes: Number of classes for classification
        train_ratio: Ratio of training samples
        val_ratio: Ratio of validation samples
        test_ratio: Ratio of test samples
    """
    # Create output directory
    output_path = Path(output_path)
    nifti_dir = output_path / "nifti_files"
    nifti_dir.mkdir(parents=True, exist_ok=True)

    # Generate random data
    all_files = []

    # Calculate splits
    num_train = int(num_files * train_ratio)
    num_val = int(num_files * val_ratio)
    num_test = num_files - num_train - num_val

    logger.info(f"Creating {num_train} training, {num_val} validation, and {num_test} test samples")

    # Create all files
    for i in range(num_files):
        # Generate random 3D volume
        data = np.random.randn(*shape).astype(np.float32)
        # Add some structure to make it look more like a real scan
        data = data + np.sin(np.linspace(0, 10, shape[0]))[:, None, None]
        data = data + np.cos(np.linspace(0, 10, shape[1]))[None, :, None]
        data = data + np.sin(np.linspace(0, 10, shape[2]))[None, None, :]

        # Create NIfTI file
        nifti_file = nifti_dir / f"sample_{i:03d}.nii.gz"
        nifti_img = nib.Nifti1Image(data, np.eye(4))
        nifti_img.to_filename(nifti_file)

        # Generate random histology labels
        histology_label = np.random.randint(0, 3)  # 0: ADC, 1: SQU, 2: SMC
        histology_adc = 1 if histology_label == 0 else 0
        histology_squ = 1 if histology_label == 1 else 0
        histology_smc = 1 if histology_label == 2 else 0

        # Generate random demographic features
        age = np.random.randint(18, 91)  # Random age between 18 and 90
        sex = np.random.randint(0, 2)    # 0 for male, 1 for female

        # Determine split
        if i < num_train:
            split = "train"
        elif i < num_train + num_val:
            split = "val"
        else:
            split = "test"

        # Add to list with labels
        all_files.append(
            {
                "image": str(nifti_file),
                "label": np.random.randint(0, num_classes),
                "histology_label": histology_label,
                "histology_adc": histology_adc,
                "histology_squ": histology_squ,
                "histology_smc": histology_smc,
                "age": age,
                "sex": sex,
                "split": split,
            }
        )

    # Create parquet file
    df = pd.DataFrame(all_files)

    # Save as parquet
    parquet_file = output_path / "dataset.parquet"
    df.to_parquet(parquet_file, index=False)

    # Also save as JSON for backward compatibility
    json_data = {
        "train": [f for f in all_files if f["split"] == "train"],
        "validation": [f for f in all_files if f["split"] == "val"],
        "test": [f for f in all_files if f["split"] == "test"],
    }
    json_file = output_path / "dummy_dataset.json"
    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=2)

    logger.info(f"Created {num_files} dummy NIfTI files in {nifti_dir}")
    logger.info(f"Created parquet file at {parquet_file}")
    logger.info(f"Created dataset JSON file at {json_file}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Training samples: {num_train}")
    logger.info(f"Validation samples: {num_val}")
    logger.info(f"Test samples: {num_test}")
    logger.info("Histology labels distribution:")
    logger.info(f"ADC: {sum(1 for x in all_files if x['histology_adc'] == 1)}")
    logger.info(f"SQU: {sum(1 for x in all_files if x['histology_squ'] == 1)}")
    logger.info(f"SMC: {sum(1 for x in all_files if x['histology_smc'] == 1)}")
    logger.info("Demographic features:")
    logger.info(f"Age range: {min(x['age'] for x in all_files)}-{max(x['age'] for x in all_files)}")
    logger.info(f"Sex distribution - Male: {sum(1 for x in all_files if x['sex'] == 0)}, Female: {sum(1 for x in all_files if x['sex'] == 1)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create dummy data for testing")
    parser.add_argument("--output_path", type=str, default="./dummy_data", help="Directory to save the dummy data")
    parser.add_argument(
        "--shape", type=int, nargs=3, default=[384, 384, 256], help="Shape of the NIfTI volumes (height width depth)"
    )
    parser.add_argument("--num_files", type=int, default=2, help="Number of NIfTI files to create")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes for classification")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of training samples")
    parser.add_argument("--val_ratio", type=float, default=0.5, help="Ratio of validation samples")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of test samples")

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
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
