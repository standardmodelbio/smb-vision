import json
import multiprocessing as mp
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Union

import boto3
import nibabel as nib

# Import the MaskGenerator and transforms from mim.py
from mim import GenerateMask, PermuteImage
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandSpatialCropSamplesd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
    Transform,
)
from tqdm import tqdm


def get_transforms(img_size=384, depth=320, mask_patch_size=32, patch_size=16, mask_ratio=0.75):
    """Create transformation pipeline"""
    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(
                keys=["image"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image", allow_smaller=False),
            RandSpatialCropSamplesd(
                keys=["image"],
                roi_size=(img_size, img_size, depth),
                random_size=False,
                num_samples=1,
            ),
            SpatialPadd(keys=["image"], spatial_size=(img_size, img_size, depth)),
        ]
    )


def verify_transforms(file_dict, transforms, temp_path):
    """Apply transforms to a single file and verify the output"""
    try:
        transformed = transforms(file_dict)

        # Check image shape
        image = transformed[0]["image"]

        # print(f"Image shape: {image.shape}")

        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        if image.shape == (1, 384, 384, 320):
            return file_dict, True
        else:
            return file_dict, False
    except Exception as e:
        print(f"Transform failed: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return file_dict, False


def process_file(s3_client, bucket, key, transforms, verify):
    try:
        # Create temp file
        temp_file = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
        temp_path = temp_file.name
        temp_file.close()

        # Download file from S3
        s3_client.download_file(bucket, key, temp_path)
        file_dict = {"image": temp_path}

        if verify:
            # print(f"\nVerifying transforms for {key}")
            result = verify_transforms(file_dict, transforms, temp_path)
            if result[1]:  # If validation successful
                # Copy to validated folder
                validated_key = "datasets/idc2niix-ct/" + "/".join(key.split("/")[2:])
                s3_client.copy_object(Bucket=bucket, CopySource={"Bucket": bucket, "Key": key}, Key=validated_key)
                # print(f"Copied validated file to: {validated_key}")
            return result
        else:
            # Just check if file can be loaded
            img = nib.load(temp_path)
            if len(img.get_fdata().shape) == 3:
                os.remove(temp_path)
                return file_dict, True
            os.remove(temp_path)
            return file_dict, False
    except Exception as e:
        print(f"Error processing {key}: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return file_dict, False


def create_dataset_json(
    s3_path="s3://smb-dev-us-east-2-data/datasets/idc2niix/",
    output_file="dataset.json",
    val_split: Union[int, float] = 0.2,
    verify=True,
):
    transforms = get_transforms() if verify else None
    s3_client = boto3.client("s3")

    # Parse S3 path
    bucket = s3_path.split("/")[2]
    prefix = "/".join(s3_path.split("/")[3:])

    # List all objects in bucket with prefix
    paginator = s3_client.get_paginator("list_objects_v2")
    files = []
    process_files = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, PaginationConfig={"MaxItems": 10}):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".nii.gz"):
                process_files.append((s3_client, bucket, obj["Key"]))

    print(f"{len(process_files)} niftis in total...")

    # Process files using ThreadPoolExecutor
    max_workers = min(32, len(process_files))  # Limit max threads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for args in process_files:
            future = executor.submit(process_file, *args, transforms=transforms, verify=verify)
            futures.append(future)

        results = []
        for future in tqdm(futures, total=len(process_files), desc="Processing files"):
            results.append(future.result())

    # Filter successful results
    files = [file_dict for file_dict, success in results if success]

    # Calculate split indices
    total = len(files)
    if isinstance(val_split, int):
        val_size = val_split
    else:
        val_size = int(total * val_split)
    train_size = total - val_size
    print(f"train_size is {train_size}, val_size is {val_size}")

    # Create dataset dict
    dataset = {"train": files[:train_size], "validation": files[train_size:]}

    # Write to json file
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=4)

    return dataset


if __name__ == "__main__":
    create_dataset_json(
        "s3://smb-dev-us-east-2-data/datasets/idc2niix/",
        output_file="./smb-vision-large-train-mim.json",
        val_split=100,
        verify=True,
    )
