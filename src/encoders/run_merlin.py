import argparse
import os
from typing import Dict, List

import pandas as pd
import torch
from base_encoder import BaseEncoder, BaseEncoderRunner
from loguru import logger
from merlin import Merlin
from tqdm import tqdm

from dataloader.load import CTDataset


class MerlinEncoder(BaseEncoder):
    """Merlin model encoder implementation"""

    def create_dataset(self, data_dict: List[Dict], args: argparse.Namespace):
        return CTDataset(data_dict, args).setup()

    def setup_model(self, image_embedding: bool = True):
        logger.info(f"Setting up Merlin model on {self.device}...")
        try:
            model = Merlin(ImageEmbedding=image_embedding)
            model.eval()
            model.to(self.device)
            logger.info("Model setup successful")
            return model
        except Exception as e:
            logger.error(f"Failed to setup model: {e}")
            raise

    def generate_embedding(self, model, image: torch.Tensor) -> torch.Tensor:
        try:
            image = image.to(self.device)
            with torch.no_grad():
                embedding = model(image)
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def save_embedding(self, embedding: torch.Tensor, uid: str, save_dir: str):
        try:
            np_embedding = embedding[0].float().cpu().numpy()
            original_shape = np_embedding.shape

            df = pd.DataFrame(
                {
                    "uid": [uid],
                    "embedding": [np_embedding.flatten()],
                    "embedding_shape": [original_shape],
                    "model_id": ["merlin"],
                }
            )

            model_dir = os.path.join(save_dir, "model_id=merlin")
            os.makedirs(model_dir, exist_ok=True)
            output_file = os.path.join(model_dir, f"{uid}.parquet")
            df.to_parquet(output_file, compression="snappy")
        except Exception as e:
            logger.error(f"Failed to save embedding: {e}")
            raise

    def process_batch(self, gpu_id, data_batch, args):
        device = torch.device(f"cuda:{gpu_id}")
        self.device = device
        model = self.setup_model(image_embedding=True)
        error_files = []

        for batch in tqdm(data_batch, desc=f"GPU {gpu_id} processing"):
            try:
                uid = batch["uid"]
                image = batch["image"]
                embedding = self.generate_embedding(model, image)
                self.save_embedding(embedding, uid, args.save_dir)
            except Exception as e:
                error_msg = f"Error processing {uid}: {str(e)}"
                logger.error(error_msg)
                error_files.append({"uid": str(uid), "error": str(e)})

        return error_files


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run encoder inference")
    parser.add_argument("--encoder", type=str, choices=["merlin", "smb-vision"], required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--saved_json_path", type=str, required=True)

    # Dataset-specific arguments
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for Merlin")
    parser.add_argument("--cache_num", type=int, default=None, help="Number of cache files")
    parser.add_argument("--cache_rate", type=float, default=None, help="Rate of cache file")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--dist", action="store_true", help="Enable distributed training")
    parser.add_argument("--bf16", action="store_true", help="Enable bfloat16 precision")

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    encoder_classes = {
        "merlin": MerlinEncoder,
        # "videomae": VideoMAEEncoder,
    }
    encoder_class = encoder_classes[args.encoder]
    runner = BaseEncoderRunner(encoder_class("cpu"))

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA devices available")

        # Setup and run
        torch.multiprocessing.set_start_method("spawn")
        dataset = runner.setup_dataset(args)
        runner.main_process_func(dataset, args)
        logger.info(f"{args.encoder} inference completed successfully")

    except Exception as e:
        logger.error(f"Fatal error in main process: {e}")
        raise


if __name__ == "__main__":
    main()
