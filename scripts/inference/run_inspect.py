import argparse
import json
import os

import awswrangler as wr
import pandas as pd
import torch
from loguru import logger

from dataloader.load import CTDataset
from transformers import VideoMAEForPreTraining


def build_json(impressions_path, image_dir, output_json_path):
    """Build a json file containing paths to nifti files from separate train/validation directories"""
    # Read impressions CSV file
    impressions_df = pd.read_csv(impressions_path)

    # Create file list with image paths
    files = []
    for _, row in impressions_df.iterrows():
        image_path = os.path.join(image_dir, f"{row['impression_id']}.nii.gz")
        files.append({"image": image_path, "uid": row["impression_id"]})

    # For this implementation, just putting all files in train
    # Could split into train/val based on labels_path if needed
    # data_dict = {"train": files, "validation": []}

    # Write to json file
    with open(output_json_path, "w") as f:
        json.dump(files, f, indent=2)

    logger.info(f"Created dataset JSON file at {output_json_path}")
    return files, output_json_path


def setup_dataset(args):
    logger.info("Building JSON file...")
    data_dict, output_json_path = build_json(args.impressions_path, args.image_dir, args.saved_json_path)

    logger.info("Samples (first 3):")
    for sample in data_dict[:3]:
        logger.info(sample)

    logger.info("Setting up dataset...")
    try:
        dataset = CTDataset(
            data_list=data_dict,
            img_size=args.img_size,
            depth=args.depth,
            batch_size=args.batch_size,
            val_batch_size=args.val_batch_size,
            num_workers=args.num_workers,
            dist=args.dist,
        )
        logger.info("Dataset setup successful")
        return dataset.setup()
    except Exception as e:
        logger.error(f"Failed to setup dataset: {e}")
        raise


def setup_model(device, model_name):
    logger.info(f"Setting up model on {device}...")
    try:
        model = VideoMAEForPreTraining.from_pretrained(model_name, trust_remote_code=True).to(device)
        logger.info("Model setup successful")
        return model
    except Exception as e:
        logger.error(f"Failed to setup model: {e}")
        raise


def generate_embedding(model, image, device):
    try:
        image = image.to(device)
        with torch.no_grad():
            embedding = model.videomae(image.unsqueeze(0))
        return embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        raise


def save_embedding(embedding, impression_id, save_path, model_id):
    try:
        np_embedding = embedding.last_hidden_state.cpu().numpy()
        df = pd.DataFrame(
            {
                "uid": [impression_id],
                "embedding": [np_embedding],
                "model_id": [model_id],
            }
        )

        wr.s3.to_parquet(
            df=df,
            path=save_path,
            dataset=True,
            partition_cols=["model_id"],
            mode="append",
            compression="snappy",
            max_rows_by_file=1000000,
        )
        logger.info(f"Saved embedding to {s3_path}")
    except Exception as e:
        logger.error(f"Failed to save embedding to {s3_path}: {e}")
        raise


def main_process_func(data, model, device, args):
    logger.info("Processing data...")
    logger.info(f"Processing {len(data)} total samples")
    error_files = []

    for i, item in enumerate(data):
        try:
            impression_id = item["uid"]
            image = item["image"]
            logger.info(f"Processing image {i + 1}/{len(data)} with shape: {image.shape}")

            embedding = generate_embedding(model, image, device)
            save_embedding(embedding, impression_id, args.save_path, args.model_name)

        except Exception as e:
            error_msg = f"Error processing {image}: {str(e)}"
            logger.error(error_msg)
            error_files.append({"file": str(image), "error": str(e)})

    if error_files:
        logger.error(f"Failed to process {len(error_files)} files")
        with open("error_files.json", "w") as f:
            json.dump(error_files, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate embeddings from medical images")
    parser.add_argument(
        "--impressions_path", type=str, default="../data/Final_impressions.csv", help="Path to dataset CSV file"
    )
    parser.add_argument("--image_dir", type=str, default="../data/asset-inspect/CTPA", help="Path to image directory")
    parser.add_argument("--saved_json_path", type=str, default="/tmp/asset-inspect.json", help="Path to JSON file")
    parser.add_argument("--img_size", type=int, default=512, help="Image size")
    parser.add_argument("--depth", type=int, default=320, help="Image depth")
    parser.add_argument("--cache_dir", type=str, default="../data/cache", help="Cache directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=1, help="Validation batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--dist", action="store_true", help="Enable distributed training")
    parser.add_argument(
        "--model_name", type=str, default="standardmodelbio/smb-vision-base-20250122", help="Model name or path"
    )
    parser.add_argument(
        "--save_path", type=str, default="s3://bucket-name/folder-name", help="Save s3 path for embeddings"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    return parser.parse_args()


if __name__ == "__main__":
    logger.info("Starting embedding generation process")
    args = parse_args()

    # Setup device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        # Setup dataset and model
        dataset = setup_dataset(args)
        model = setup_model(device, args.model_name)

        # Process train and validation splits
        main_process_func(dataset, model, device, args)

        logger.info("Embedding generation process completed successfully")

    except Exception as e:
        logger.error(f"Fatal error in main process: {e}")
        raise
