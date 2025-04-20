# Standard library imports
import argparse
import json
import os

# Third-party library imports
import awswrangler as wr
import pandas as pd
import torch
from loguru import logger

# Local imports
from dataloader.load import CTDataset
from models.videomae.modeling_videomae import VideoMAEForPreTraining


def build_json(impressions_path, image_dir, output_json_path):
    """Build a json file containing paths to nifti files from separate train/validation directories

    Args:
        impressions_path (str): Path to CSV file containing image impressions
        image_dir (str): Directory containing the image files
        output_json_path (str): Path to save the output JSON file

    Returns:
        tuple: List of files and path to created JSON
    """
    # Read impressions CSV file
    impressions_df = pd.read_csv(impressions_path)

    # Create file list with image paths
    files = []
    missing_files = []
    for _, row in impressions_df.iterrows():
        image_path = os.path.join(image_dir, f"{row['impression_id']}.nii.gz")
        if os.path.exists(image_path):
            files.append({"image": image_path, "uid": row["impression_id"]})
        else:
            missing_files.append(image_path)
            logger.warning(f"Image file not found: {image_path}")

    if missing_files:
        logger.warning(f"Total missing files: {len(missing_files)}")

    # For this implementation, just putting all files in train
    # Could split into train/val based on labels_path if needed
    # data_dict = {"train": files, "validation": []}

    # Write to json file
    with open(output_json_path, "w") as f:
        json.dump(files, f, indent=2)

    logger.info(f"Created dataset JSON file at {output_json_path}")
    return files, output_json_path


def setup_dataset(args):
    """Set up the dataset for training

    Args:
        args: Command line arguments

    Returns:
        CTDataset: The configured dataset
    """
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
            bf16=args.bf16,
        )
        logger.info("Dataset setup successful")
        return dataset.setup()
    except Exception as e:
        logger.error(f"Failed to setup dataset: {e}")
        raise


def setup_model(device, model_name):
    """Initialize the VideoMAE model

    Args:
        device: PyTorch device
        model_name (str): Name or path of pretrained model

    Returns:
        VideoMAEForPreTraining: The initialized model
    """
    logger.info(f"Setting up model on {device}...")
    try:
        model = VideoMAEForPreTraining.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", trust_remote_code=True
        ).to(device)
        logger.info("Model setup successful")
        return model
    except Exception as e:
        logger.error(f"Failed to setup model: {e}")
        raise


def generate_embedding(model, image, device):
    """Generate embeddings from input image

    Args:
        model: The VideoMAE model
        image: Input image tensor
        device: PyTorch device

    Returns:
        torch.Tensor: Generated embedding
    """
    try:
        image = image.to(device)
        with torch.no_grad():
            embedding = model.videomae(image.unsqueeze(0))
        return embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        raise


def save_embedding(embedding, impression_id, save_path, model_id):
    """Save generated embedding to S3

    Args:
        embedding: The embedding tensor
        impression_id: Unique identifier for the image
        save_path (str): S3 path to save embeddings
        model_id (str): Identifier for the model used
    """
    try:
        np_embedding = embedding.last_hidden_state.squeeze(0).float().cpu().numpy()
        # Store original shape before flattening
        original_shape = np_embedding.shape
        # Convert to nested list before storing in DataFrame
        df = pd.DataFrame(
            {
                "uid": [impression_id],
                "embedding": [np_embedding.flatten()],
                "embedding_shape": [original_shape],
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
        logger.info(f"Saved embedding to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save embedding to {save_path}: {e}")
        raise


def main_process_func(data, model, device, args):
    """Main processing function to generate and save embeddings

    Args:
        data: Dataset containing images
        model: The VideoMAE model
        device: PyTorch device
        args: Command line arguments
    """
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
            error_msg = f"Error processing {impression_id}: {str(e)}"
            logger.error(error_msg)
            error_files.append({"uid": str(impression_id), "error": str(e)})

    if error_files:
        logger.error(f"Failed to process {len(error_files)} files")
        with open("error_files.json", "w") as f:
            json.dump(error_files, f, indent=2)


def parse_args():
    """Parse command line arguments

    Returns:
        argparse.Namespace: Parsed arguments
    """
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
    parser.add_argument("--bf16", action="store_true", help="Enable bfloat16 precision")
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
