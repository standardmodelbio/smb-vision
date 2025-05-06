import argparse
import os
from typing import Dict, List, Any

import pandas as pd
import torch
from base_encoder import BaseEncoder, BaseEncoderRunner
from loguru import logger
from transformers import SiglipProcessor, SiglipVisionModel
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from dataloader.load import SiglipDataset


class SiglipEncoder(BaseEncoder):
    """SigLIP model encoder implementation for 2D x-ray images"""

    def __init__(self, model_id: str, device: str):
        super().__init__(device)
        self.model_id = model_id
        self.model = None
        self.processor = None

    def create_dataset(self, data_dict: List[Dict], args: argparse.Namespace) -> Dataset:
        """Create and return the dataset.

        Args:
            data_dict (List[Dict]): List of dictionaries containing image data
            args (argparse.Namespace): Command line arguments

        Returns:
            Dataset: The created dataset
        """
        return SiglipDataset(self.model_id, data_dict, cache_dir=args.cache_dir)

    def setup_model(self, image_embedding: bool = True):
        """Setup the SigLIP model for inference.

        Args:
            image_embedding (bool): Whether to setup for image embedding

        Returns:
            Tuple[SiglipVisionModel, Optional[SiglipProcessor]]: The model and optionally the processor
        """
        if self.model is None:
            logger.info(f"Setting up SigLIP model on {self.device}...")
            try:
                self.model = SiglipVisionModel.from_pretrained(
                    f"google/{self.model_id}",
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16,
                )
                self.model.eval()
                self.model.to(self.device)

                if image_embedding:
                    self.processor = SiglipProcessor.from_pretrained(f"google/{self.model_id}")

                logger.info("Model setup successful")
            except Exception as e:
                logger.error(f"Failed to setup model: {e}")
                raise

        return self.model, self.processor

    def generate_embedding(self, model, images: torch.Tensor) -> torch.Tensor:
        """Generate embeddings for a batch of images.

        Args:
            model: The SigLIP model
            images (torch.Tensor): Batch of images to process

        Returns:
            torch.Tensor: Generated embeddings
        """
        try:
            images = images.to(self.device)
            with torch.inference_mode():
                outputs = model(images, output_hidden_states=True)
                embeddings = outputs.last_hidden_state
                # print(embeddings.shape)
                # embeddings = outputs / outputs.norm(dim=-1, keepdim=True)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def save_embedding(self, embeddings: torch.Tensor, uids: List[str], save_dir: str, model_id: str):
        """Save generated embeddings.

        Args:
            embeddings (torch.Tensor): Batch of embeddings to save
            uids (List[str]): List of UIDs corresponding to the embeddings
            save_dir (str): Directory to save the embeddings
            model_id (str): ID of the model used to generate embeddings
        """
        try:
            np_embeddings = embeddings.cpu().float().numpy()
            model_dir = os.path.join(save_dir, f"model_id={model_id}")
            os.makedirs(model_dir, exist_ok=True)

            for i, uid in enumerate(uids):
                embedding = np_embeddings[i]
                original_shape = embedding.shape

                df = pd.DataFrame(
                    {
                        "uid": [uid],
                        "embedding": [embedding.flatten()],
                        "embedding_shape": [original_shape],
                        "model_id": [model_id],
                    }
                )

                output_file = os.path.join(model_dir, f"{uid}.parquet")
                df.to_parquet(output_file, compression="snappy")
        except Exception as e:
            logger.error(f"Failed to save embedding: {e}")
            raise

    def process_batch(self, gpu_id: int, batch: Dict[str, Any], args: argparse.Namespace) -> List[Dict]:
        """Process a single batch of data.

        Args:
            gpu_id (int): ID of the GPU to use
            batch (Dict[str, Any]): Batch of data to process
            args (argparse.Namespace): Command line arguments

        Returns:
            List[Dict]: List of errors encountered during processing
        """
        device = torch.device(f"cuda:{gpu_id}")
        self.device = device
        model, _ = self.setup_model(image_embedding=True)
        error_files = []

        try:
            uids = batch["uid"]
            images = batch["image"]

            embeddings = self.generate_embedding(model, images)
            self.save_embedding(embeddings, uids, args.save_dir, args.model_id)
        except Exception as e:
            error_msg = f"Error processing batch: {str(e)}"
            logger.error(error_msg)
            error_files.append({"error": str(e)})

        return error_files


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run SigLIP encoder inference")
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--input_json", type=str, required=True, help="Path to JSON file containing image paths")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for SigLIP")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--dist", action="store_true", help="Enable distributed training")
    parser.add_argument("--bf16", action="store_true", help="Enable bfloat16 precision")

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    runner = BaseEncoderRunner(SiglipEncoder(args.model_id, "cpu"))

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA devices available")

        # Setup and run
        torch.multiprocessing.set_start_method("spawn")
        dataset = runner.setup_dataset(args)
        runner.main_process_func(dataset, args)
        logger.info(f"{args.model_id} inference completed successfully")

    except Exception as e:
        logger.error(f"Fatal error in main process: {e}")
        raise


if __name__ == "__main__":
    main()
