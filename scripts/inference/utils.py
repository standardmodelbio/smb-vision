import json
import os

from loguru import logger
from tqdm import tqdm


def build_json(image_dir, save_dir, output_json_path):
    """Build a json file containing paths to nifti files and track processed files

    Args:
        image_dir (str): Directory containing the nifti image files
        save_dir (str): Directory where processed embeddings are saved
        output_json_path (str): Path to save the output JSON file

    Returns:
        tuple: List of unprocessed files and path to created JSON file
    """
    files = []
    processed_uids = set()

    # Read processed UIDs from parquet files in save_dir
    if os.path.exists(save_dir):
        for model_dir in os.listdir(save_dir):
            model_path = os.path.join(save_dir, model_dir)
            if os.path.isdir(model_path):
                for parquet_file in os.listdir(model_path):
                    if parquet_file.endswith(".parquet"):
                        processed_uids.add(parquet_file.replace(".parquet", ""))
        logger.info(f"Found {len(processed_uids)} previously processed UIDs from parquet files")

    # Read files from image_dir
    for filename in tqdm(os.listdir(image_dir), desc="Building file list"):
        if filename.endswith(".nii.gz"):
            uid = filename.replace(".nii.gz", "")
            if uid not in processed_uids:
                image_path = os.path.join(image_dir, filename)
                files.append({"image": image_path, "uid": uid})

    # Write to json file
    with open(output_json_path, "w") as f:
        json.dump(files, f, indent=2)

    logger.info(f"Created/updated dataset JSON file at {output_json_path} with {len(files)} unprocessed files")
    return files, output_json_path
