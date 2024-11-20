import json
import os
from typing import Union

import boto3


def create_dataset_json(data_dir, output_file="dataset.json", val_split: Union[int, float] = 0.2):
    files = []

    if data_dir.startswith("s3://"):
        # Parse S3 path
        bucket = data_dir.split("/")[2]
        prefix = "/".join(data_dir.split("/")[3:])

        # Initialize S3 client
        s3 = boto3.client("s3")

        # List objects in S3 bucket
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if "Contents" in page:
                for obj in page["Contents"]:
                    if obj["Key"].endswith(".nii.gz"):
                        file_path = f"s3://{bucket}/{obj['Key']}"
                        files.append({"image": file_path})
    else:
        for root, dirs, filenames in os.walk(data_dir):
            for filename in filenames:
                if filename.endswith(".nii.gz"):
                    file_path = os.path.join(root, filename)
                    files.append({"image": file_path})

    # Calculate split indices
    total = len(files)
    if isinstance(val_split, int):
        val_size = val_split
    else:
        val_size = int(total * val_split)
    train_size = total - val_size
    print(total, train_size, val_size)

    # Create dataset dict
    dataset = {"train": files[:train_size], "validation": files[train_size:]}

    # Write to json file
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=4)

    return dataset


if __name__ == "__main__":
    create_dataset_json("../nifti_files", output_file="./smb-vision-large-train-mim.json", val_split=100)
