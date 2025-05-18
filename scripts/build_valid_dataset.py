import argparse
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import torch
from loguru import logger
from tqdm import tqdm

from dataloader.load import MIMDataset


def collect_nifti_files(directory_path: str) -> List[Dict[str, str]]:
    """Collect all NIfTI files in directory and subdirectories.

    Args:
        directory_path (str): Directory containing NIfTI files

    Returns:
        List[Dict[str, str]]: List of dictionaries with image paths
    """
    nifti_files = []
    for ext in tqdm(["*.nii", "*.nii.gz"], desc="Searching for files"):
        paths = glob.glob(os.path.join(directory_path, "**", ext), recursive=True)
        nifti_files.extend([{"image": path} for path in paths])

    logger.info(f"Found {len(nifti_files)} NIfTI files")
    return nifti_files


def validate_single_file(args: Tuple[Dict[str, str], MIMDataset, int]) -> Tuple[Dict[str, str], bool]:
    """Validate a single file using MIMDataset.

    Args:
        args (Tuple[Dict[str, str], MIMDataset, int]): Tuple containing (file_dict, dataset, index)

    Returns:
        Tuple[Dict[str, str], bool]: Tuple of (file_dict, is_valid)
    """
    file_dict, dataset, idx = args
    try:
        # Try to load and transform the image
        sample = dataset[idx]
        if isinstance(sample["image"], torch.Tensor):
            return file_dict, True
    except Exception as e:
        logger.warning(f"Error processing file {file_dict['image']}: {str(e)}")
    return file_dict, False


def validate_files(
    data_list: List[Dict[str, str]], cache_dir: str
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Validate files using MIMDataset and split into train/val sets.

    Args:
        data_list (List[Dict[str, str]]): List of dictionaries with image paths
        cache_dir (str): Directory for caching processed images

    Returns:
        Tuple[List[Dict[str, str]], List[Dict[str, str]]]: Valid train and validation files
    """
    # Create temporary dataset for validation
    dataset = MIMDataset(data=data_list, cache_dir=cache_dir)

    # Prepare arguments for parallel processing
    validation_args = [(data_list[i], dataset, i) for i in range(len(dataset))]

    # Use ThreadPoolExecutor for parallel validation
    valid_files = []
    max_workers = min(32, len(data_list))  # Limit max threads to avoid memory issues

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all validation tasks
        future_to_args = {executor.submit(validate_single_file, args): args for args in validation_args}

        # Process results as they complete with progress bar
        pbar = tqdm(
            total=len(validation_args),
            desc="Validating files",
            unit="files",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        for future in as_completed(future_to_args):
            file_dict, is_valid = future.result()
            if is_valid:
                valid_files.append(file_dict)
            pbar.update(1)
            pbar.set_postfix({"valid": len(valid_files)})
        pbar.close()

    # Split into train/val sets (80/20 split)
    split_idx = int(len(valid_files) * 0.9)
    train_files = valid_files[:split_idx]
    val_files = valid_files[split_idx:]

    logger.info(f"Found {len(valid_files)} valid files")
    logger.info(f"Training set: {len(train_files)} files")
    logger.info(f"Validation set: {len(val_files)} files")

    return train_files, val_files


def main():
    parser = argparse.ArgumentParser(description="Build and validate dataset JSON file")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing NIfTI files")
    parser.add_argument("--output_json", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--cache_dir", type=str, required=True, help="Directory for caching processed images")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of worker threads for validation")
    args = parser.parse_args()

    # Create cache directory if it doesn't exist
    os.makedirs(args.cache_dir, exist_ok=True)

    # Collect NIfTI files
    data_list = collect_nifti_files(args.input_dir)

    # Validate files and split into train/val sets
    train_files, val_files = validate_files(data_list, args.cache_dir)

    # Create dataset dictionary
    dataset = {"train": train_files, "validation": val_files}

    # Save to JSON file
    with open(args.output_json, "w") as f:
        json.dump(dataset, f, indent=2)

    logger.info(f"Saved dataset to {args.output_json}")


if __name__ == "__main__":
    main()
