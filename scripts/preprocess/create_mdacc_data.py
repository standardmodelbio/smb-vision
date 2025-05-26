import json
import sys
from pathlib import Path

import pandas as pd
from loguru import logger


def create_mdanderson_dataset(
    data_path: str,
    train_image_dir: str,
    val_image_dir: str,
    output_path: str,
    test_ratio: float = 0.15,
):
    """
    Create dataset from MD Anderson data with survival information.

    Args:
        data_path: Path to the Excel/CSV file containing survival data
        train_image_dir: Directory containing the training NIfTI files
        val_image_dir: Directory containing the validation NIfTI files
        output_path: Directory to save the processed dataset
        test_ratio: Ratio of test samples (will be taken from validation set)
    """
    # Create output directory
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Read the survival data
    if data_path.endswith(".xlsx"):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)

    # Create dataset entries
    all_files = []
    for _, row in df.iterrows():
        patient_id = row["PatientID"]
        patient_num = patient_id.split("-")[1]

        # Try to find image in train directory first
        train_image_path = Path(train_image_dir) / f"CT_Main_{patient_num}.nii.gz"
        val_image_path = Path(val_image_dir) / f"CT_Main_{patient_num}.nii.gz"

        if train_image_path.exists():
            image_path = train_image_path
            split = "train"
        elif val_image_path.exists():
            image_path = val_image_path
            # We'll determine if this goes to val or test later
            split = "val"
        else:
            logger.warning(f"Image not found for patient {patient_id} in either train or val directories")
            continue

        # Add to list with labels
        all_files.append(
            {
                "image": str(image_path),
                "patient_id": patient_id,
                "os": float(row["OS"]),
                "os_event": int(row["OS_events"]),
                "split": split,
            }
        )

    # Calculate number of test samples from validation set
    val_samples = [f for f in all_files if f["split"] == "val"]
    num_test = int(len(val_samples) * test_ratio)

    # Randomly select validation samples to move to test set
    import random

    random.seed(42)  # For reproducibility
    test_indices = random.sample(range(len(val_samples)), num_test)

    # Update splits
    for i, idx in enumerate(test_indices):
        val_samples[idx]["split"] = "test"

    # Create parquet file
    df_out = pd.DataFrame(all_files)
    parquet_file = output_path / "mdanderson_dataset.parquet"
    df_out.to_parquet(parquet_file, index=False)

    # Also save as JSON for backward compatibility
    json_data = {
        "train": [f for f in all_files if f["split"] == "train"],
        "validation": [f for f in all_files if f["split"] == "val"],
        "test": [f for f in all_files if f["split"] == "test"],
    }
    json_file = output_path / "mdacc_dataset.json"
    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=2)

    # Log statistics
    num_train = len([f for f in all_files if f["split"] == "train"])
    num_val = len([f for f in all_files if f["split"] == "val"])
    num_test = len([f for f in all_files if f["split"] == "test"])

    logger.info(f"Created parquet file at {parquet_file}")
    logger.info(f"Created dataset JSON file at {json_file}")
    logger.info(f"Total samples: {len(all_files)}")
    logger.info(f"Training samples: {num_train}")
    logger.info(f"Validation samples: {num_val}")
    logger.info(f"Test samples: {num_test}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create MD Anderson dataset")
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the Excel/CSV file containing survival data"
    )
    parser.add_argument(
        "--train_image_dir", type=str, required=True, help="Directory containing the training NIfTI files"
    )
    parser.add_argument(
        "--val_image_dir", type=str, required=True, help="Directory containing the validation NIfTI files"
    )
    parser.add_argument(
        "--output_path", type=str, default="./mdanderson_data", help="Directory to save the processed dataset"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.15, help="Ratio of test samples (taken from validation set)"
    )

    args = parser.parse_args()

    # Configure logger
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )

    create_mdanderson_dataset(
        data_path=args.data_path,
        train_image_dir=args.train_image_dir,
        val_image_dir=args.val_image_dir,
        output_path=args.output_path,
        test_ratio=args.test_ratio,
    )
