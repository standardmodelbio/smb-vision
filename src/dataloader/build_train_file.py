import json
import os
from typing import Union

import nibabel as nib
from loguru import logger


def create_dataset_json(data_dir, output_file="dataset.json", val_split: Union[int, float] = 0.2):
    files = []

    for root, dirs, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename.endswith(".nii.gz"):
                file_path = os.path.join(root, filename)
                try:
                    # Try to load the nifti file
                    nib.load(file_path)
                    files.append({"image": file_path})
                except:
                    logger.info(f"Corrupted nifti file: {file_path}")

    # Calculate split indices
    total = len(files)
    if isinstance(val_split, int):
        val_size = val_split
    else:
        val_size = int(total * val_split)
    train_size = total - val_size
    logger.info(f"train_size is {train_size}, val_size is {val_size}")

    # Create dataset dict
    dataset = {"train": files[:train_size], "validation": files[train_size:]}

    # Write to json file
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=4)

    return dataset


if __name__ == "__main__":
    create_dataset_json("../nifti_files", output_file="./smb-vision-large-train-mim.json", val_split=100)
