import argparse
import os
from typing import Dict, List

import pandas as pd
import torch
from base_encoder import BaseEncoder, BaseEncoderRunner
from loguru import logger
from transformers import SiglipProcessor, SiglipModel
from tqdm import tqdm

from dataloader.load import SiglipDataset


class SiglipEncoder(BaseEncoder):
    """SigLIP model encoder implementation for 2D x-ray images"""

    def __init__(self, device: str):
        super().__init__(device)
        self.model_id = "siglip"  # Add model_id for the base class to use

    def create_dataset(self, data_dict: List[Dict], args: argparse.Namespace):
        return SiglipDataset(data_dict, cache_dir=args.cache_dir)

    def setup_model(self, image_embedding: bool = True):
        logger.info(f"Setting up SigLIP model on {self.device}...")
        try:
            processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")
            model = SiglipModel.from_pretrained("google/siglip-base-patch16-224")
            model.eval()
            model.to(self.device)
            logger.info("Model setup successful")
            return model, processor
        except Exception as e:
            logger.error(f"Failed to setup model: {e}")
            raise

    def generate_embedding(self, model, image: torch.Tensor) -> torch.Tensor:
        try:
            image = image.unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = model.get_image_features(image)
                embedding = outputs / outputs.norm(dim=-1, keepdim=True)
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def save_embedding(self, embedding: torch.Tensor, uid: str, save_dir: str, model_id: str):
        try:
            np_embedding = embedding.cpu().numpy()
            original_shape = np_embedding.shape

            df = pd.DataFrame(
                {
                    "uid": [uid],
                    "embedding": [np_embedding.flatten()],
                    "embedding_shape": [original_shape],
                    "model_id": [model_id],
                }
            )

            model_dir = os.path.join(save_dir, f"model_id={model_id}")
            os.makedirs(model_dir, exist_ok=True)
            output_file = os.path.join(model_dir, f"{uid}.parquet")
            df.to_parquet(output_file, compression="snappy")
        except Exception as e:
            logger.error(f"Failed to save embedding: {e}")
            raise

    def process_batch(self, gpu_id, data_batch, args):
        device = torch.device(f"cuda:{gpu_id}")
        self.device = device
        model, processor = self.setup_model(image_embedding=True)
        error_files = []

        for batch in tqdm(data_batch, desc=f"GPU {gpu_id} processing"):
            try:
                uid = batch["uid"]
                image = batch["image"]
                # Process image through SigLIP processor
                inputs = processor(images=image, return_tensors="pt")
                image = inputs.pixel_values.squeeze(0)
                embedding = self.generate_embedding(model, image)
                self.save_embedding(embedding, uid, args.save_dir, args.model_id)
            except Exception as e:
                error_msg = f"Error processing {uid}: {str(e)}"
                logger.error(error_msg)
                error_files.append({"uid": str(uid), "error": str(e)})

        return error_files


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run SigLIP encoder inference")
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--input_json", type=str, required=True, help="Path to JSON file containing image paths")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for SigLIP")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--dist", action="store_true", help="Enable distributed training")
    parser.add_argument("--bf16", action="store_true", help="Enable bfloat16 precision")

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    encoder_classes = {
        "siglip": SiglipEncoder,
    }
    encoder_class = encoder_classes[args.model_id]
    runner = BaseEncoderRunner(encoder_class("cpu"))

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
