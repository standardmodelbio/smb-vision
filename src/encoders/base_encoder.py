import argparse
import json
import multiprocessing as mp
import os
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Tuple

import torch
from loguru import logger
from torch.utils.data import Dataset
from tqdm import tqdm


class BaseEncoder(ABC):
    """Base class for all encoders"""

    FILE_EXTENSION = ".nii.gz"  # Adjust based on your input format

    def __init__(self, device: str):
        self.device = device

    @abstractmethod
    def create_dataset(self, data_dict: List[Dict], args: argparse.Namespace) -> Dataset:
        """Initialize the dataset"""
        pass

    @abstractmethod
    def setup_model(self, **kwargs) -> Any:
        """Initialize the model"""
        pass

    @abstractmethod
    def generate_embedding(self, model: Any, image: torch.Tensor) -> torch.Tensor:
        """Generate embeddings from input image"""
        pass

    @abstractmethod
    def save_embedding(self, embedding: torch.Tensor, uid: str, save_dir: str, model_id: str, **kwargs):
        """Save generated embedding"""
        pass

    @abstractmethod
    def process_batch(self, gpu_id: int, data_batch: List, args: argparse.Namespace) -> List[Dict]:
        """Process a batch of data"""
        pass


class BaseEncoderRunner:
    """Base class for running encoders"""

    def __init__(self, encoder_class: BaseEncoder):
        self.encoder_class = encoder_class

    def build_json(self, image_dir: str, save_dir: str, output_json_path: str) -> Tuple[List, str]:
        """Build a json file containing paths to files and track processed files"""
        files = []
        processed_uids = set()

        if os.path.exists(save_dir):
            for model_dir in os.listdir(save_dir):
                model_path = os.path.join(save_dir, model_dir)
                if os.path.isdir(model_path):
                    for parquet_file in os.listdir(model_path):
                        if parquet_file.endswith(".parquet"):
                            processed_uids.add(parquet_file.replace(".parquet", ""))
            logger.info(f"Found {len(processed_uids)} previously processed UIDs")

        for filename in tqdm(os.listdir(image_dir), desc="Building file list"):
            if filename.endswith(self.encoder_class.FILE_EXTENSION):
                uid = filename.replace(self.encoder_class.FILE_EXTENSION, "")
                if uid not in processed_uids:
                    image_path = os.path.join(image_dir, filename)
                    files.append({"image": image_path, "uid": uid})

        with open(output_json_path, "w") as f:
            json.dump(files, f, indent=2)

        logger.info(f"Created/updated dataset JSON file with {len(files)} unprocessed files")
        return files, output_json_path

    def setup_dataset(self, args: argparse.Namespace) -> Dataset:
        """Set up the dataset for processing"""
        logger.info("Building JSON file...")
        data_dict, _ = self.build_json(
            args.image_dir, os.path.join(args.save_dir, f"model_id={args.model_id}"), args.saved_json_path
        )

        logger.info("Samples (first 3):")
        for sample in data_dict[:3]:
            logger.info(sample)

        logger.info("Setting up dataset...")
        try:
            dataset = self.encoder_class.create_dataset(data_dict, args)
            logger.info("Dataset setup successful")
            return dataset
        except Exception as e:
            logger.error(f"Failed to setup dataset: {e}")
            raise

    def main_process_func(self, data: Dataset, args: argparse.Namespace):
        """Main processing function using multiple GPUs"""
        logger.info("Processing data...")
        num_gpus = torch.cuda.device_count()
        logger.info(f"Using {num_gpus} GPUs")

        # Split data into chunks for each GPU
        chunk_size = len(data) // num_gpus + (1 if len(data) % num_gpus else 0)
        data_chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

        # Create process pool
        pool = mp.Pool(processes=num_gpus)
        process_func = partial(self.encoder_class.process_batch, args=args)

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
