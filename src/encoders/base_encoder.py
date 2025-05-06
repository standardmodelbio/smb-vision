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

    def load_input_json(self, input_json_path: str) -> List[Dict]:
        """Load and validate the input JSON file"""
        try:
            with open(input_json_path, "r") as f:
                data = json.load(f)

            if not isinstance(data, dict) or "images" not in data:
                raise ValueError("JSON file must contain an 'images' key with a list of image data")

            if not isinstance(data["images"], list):
                raise ValueError("'images' must be a list")

            return data["images"]
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error loading JSON file: {str(e)}")

    def filter_processed_files(self, files: List[Dict], save_dir: str) -> List[Dict]:
        """Filter out already processed files"""
        processed_uids = set()
        model_save_dir = os.path.join(save_dir, f"model_id={self.encoder_class.model_id}")

        if os.path.exists(model_save_dir):
            for parquet_file in os.listdir(model_save_dir):
                if parquet_file.endswith(".parquet"):
                    processed_uids.add(parquet_file.replace(".parquet", ""))
            logger.info(f"Found {len(processed_uids)} previously processed UIDs")

        unprocessed_files = [f for f in files if f["uid"] not in processed_uids]
        logger.info(f"Found {len(unprocessed_files)} unprocessed files out of {len(files)} total files")
        return unprocessed_files

    def setup_dataset(self, args: argparse.Namespace) -> Dataset:
        """Set up the dataset for processing"""
        logger.info("Loading input JSON file...")
        try:
            data_dict = self.load_input_json(args.input_json)
            logger.info(f"Loaded {len(data_dict)} images from input JSON")
        except Exception as e:
            logger.error(f"Failed to load input JSON: {e}")
            raise

        # Filter out already processed files
        data_dict = self.filter_processed_files(data_dict, args.save_dir)

        if not data_dict:
            logger.warning("No unprocessed files found. Exiting.")
            return None

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
        if data is None:
            logger.info("No data to process. Exiting.")
            return

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
