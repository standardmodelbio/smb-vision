import glob
import json
import os

import torch
from safetensors.torch import save_file
from transformers import VideoMAEForPreTraining

from dataloader.load import CTDataset


def build_json_from_nifti_files(train_dir, val_dir, output_json_path):
    """Build a json file containing paths to nifti files from separate train/validation directories"""
    # Find all .nii or .nii.gz files in train directory
    train_files = []
    for ext in ["*.nii", "*.nii.gz"]:
        paths = glob.glob(os.path.join(train_dir, "**", ext), recursive=True)
        train_files.extend([{"image": path} for path in paths])

    # Find all .nii or .nii.gz files in validation directory
    val_files = []
    for ext in ["*.nii", "*.nii.gz"]:
        paths = glob.glob(os.path.join(val_dir, "**", ext), recursive=True)
        val_files.extend([{"image": path} for path in paths])

    data_dict = {"train": train_files, "validation": val_files}

    # Write to json file
    with open(output_json_path, "w") as f:
        json.dump(data_dict, f, indent=2)

    return output_json_path


if __name__ == "__main__":
    # Build json file of dataset paths
    # train_dir = "../mdanderson/public_data/train/CT/"
    # val_dir = "../mdanderson/public_data/valid/CT/"
    # json_path = "../data/dataset.json"
    # build_json_from_nifti_files(train_dir, val_dir, json_path)

    # Example usage/testing of CTDataset
    dataset = CTDataset(
        json_path="../data/dataset.json",
        img_size=384,
        depth=320,
        downsample_ratio=[1.0, 1.0, 1.0],
        cache_dir="../data/cache",
        batch_size=1,
        val_batch_size=1,
        num_workers=4,
        dist=False,
    )
    data = dataset.setup("train")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pre-trained model
    model = VideoMAEForPreTraining.from_pretrained("standardmodelbio/smb-vision-base", trust_remote_code=True).to(
        device
    )

    # Create output directory for embeddings
    os.makedirs("embeddings", exist_ok=True)

    # Process each batch
    for batch in dataset.train_dataloader(data["train"]):
        image = batch["image"]
        filepath = batch["image_meta_dict"]["filename_or_obj"][0]
        save_name = os.path.splitext(os.path.basename(filepath))[0]

        # Move image to device and generate embeddings
        image = image.to(device)
        with torch.no_grad():
            embedding = model.videomae(image)

        # Save embeddings with corresponding filepath
        tensors = {"embedding": embedding.last_hidden_state}
        save_file(tensors, f"embeddings/{save_name}.safetensors")

        break
