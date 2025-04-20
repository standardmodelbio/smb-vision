# Standard library imports
import argparse
import json
import multiprocessing as mp
import os
from functools import partial

# Third-party library imports
import awswrangler as wr
import boto3
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

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
    files = []

    # If output_json_path exists, read existing files
    existing_files = set()
    if os.path.exists(output_json_path):
        with open(output_json_path, "r") as f:
            existing_data = json.load(f)
            for item in existing_data:
                existing_files.add(item["uid"])
            # files.extend(existing_data)

    # Read files from image_dir
    for filename in tqdm(os.listdir(image_dir), desc="Building file list"):
        if filename.endswith(".nii.gz"):
            uid = filename.replace(".nii.gz", "")
            if uid not in existing_files:
                image_path = os.path.join(image_dir, filename)
                files.append({"image": image_path, "uid": uid})

    # Write to json file
    with open(output_json_path, "w") as f:
        json.dump(files, f, indent=2)

    logger.info(f"Created/updated dataset JSON file at {output_json_path}")
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


def setup_model(model_name, device):
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
    except Exception as e:
        logger.error(f"Failed to save embedding to {save_path}: {e}")
        raise


def process_batch(gpu_id, data_batch, args):
    """Process a batch of data on specified GPU

    Args:
        gpu_id: GPU device ID
        data_batch: Batch of data to process
        args: Command line arguments
    """
    device = torch.device(f"cuda:{gpu_id}")
    model = setup_model(args.model_name, device)
    model_id = args.model_name.split("/")[-1]
    error_files = []

    for item in tqdm(data_batch, desc=f"GPU {gpu_id} processing"):
        try:
            impression_id = item["uid"]
            image = item["image"]
            embedding = generate_embedding(model, image, device)
            save_embedding(embedding, impression_id, args.save_path, model_id)

        except Exception as e:
            error_msg = f"Error processing {impression_id}: {str(e)}"
            logger.error(error_msg)
            error_files.append({"uid": str(impression_id), "error": str(e)})

    return error_files


def main_process_func(data, args):
    """Main processing function using multiple GPUs

    Args:
        data: Dataset containing images
        args: Command line arguments
    """
    logger.info("Processing data...")
    num_gpus = torch.cuda.device_count()
    logger.info(f"Using {num_gpus} GPUs")

    # Split data into chunks for each GPU
    chunk_size = len(data) // num_gpus
    if len(data) % num_gpus != 0:
        chunk_size += 1
    data_chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

    # Create process pool
    pool = mp.Pool(processes=num_gpus)
    process_func = partial(process_batch, args=args)

    # Process data in parallel
    error_files = []
    try:
        for result in tqdm(
            pool.starmap(process_func, enumerate(data_chunks)), total=len(data_chunks), desc="Processing batches"
        ):
            error_files.extend(result)
    finally:
        pool.close()
        pool.join()

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
    parser.add_argument("--saved_json_path", type=str, default="../data/asset-inspect.json", help="Path to JSON file")
    parser.add_argument("--img_size", type=int, default=512, help="Image size")
    parser.add_argument("--depth", type=int, default=320, help="Image depth")
    parser.add_argument("--cache_dir", type=str, default="../data/cache", help="Cache directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=1, help="Validation batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loading workers")
    parser.add_argument("--dist", action="store_true", help="Enable distributed training")
    parser.add_argument(
        "--model_name", type=str, default="standardmodelbio/smb-vision-base-20250122", help="Model name or path"
    )
    parser.add_argument(
        "--save_path", type=str, default="s3://bucket-name/folder-name", help="Save s3 path for embeddings"
    )
    parser.add_argument("--bf16", action="store_true", help="Enable bfloat16 precision")
    return parser.parse_args()


if __name__ == "__main__":
    # Use your AWS SSO profile directly
    # for profile in boto3.session.Session().available_profiles:
    #     print(profile)
    boto3.setup_default_session(region_name="us-east-2")

    # Set multiprocessing start method to 'spawn'
    torch.multiprocessing.set_start_method("spawn")

    logger.info("Starting embedding generation process")
    args = parse_args()

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA devices available")

        # Setup dataset
        dataset = setup_dataset(args)

        # Process using multiple GPUs
        main_process_func(dataset, args)

        logger.info("Embedding generation process completed successfully")

    except Exception as e:
        logger.error(f"Fatal error in main process: {e}")
        raise
