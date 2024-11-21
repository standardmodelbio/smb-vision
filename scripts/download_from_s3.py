import json
import os

import boto3


def download_nifti_from_s3(json_file):
    # Initialize S3 client
    s3_client = boto3.client("s3")

    # Read JSON file
    with open(json_file) as f:
        data = json.load(f)

    # Download each nifti file
    for sample in data["train"] + data["validation"]:
        path = sample["image"]
        bucket_name = "smb-dev-us-east-2-data"
        key = "datasets/idc2niix/" + "/".join(path.split("/")[2:])
        local_file = "../nifti_files/" + "/".join(path.split("/")[2:])

        # Create local directory if needed
        os.makedirs(os.path.dirname(local_file), exist_ok=True)

        # Download file
        try:
            s3_client.download_file(bucket_name, key, local_file)
            print(f"Downloaded {path}")
        except Exception as e:
            print(f"Error downloading {path}: {e}")


if __name__ == "__main__":
    # Parse command line arguments
    json_file = "./smb-vision-large-train-mim.json"
    download_nifti_from_s3(json_file)
