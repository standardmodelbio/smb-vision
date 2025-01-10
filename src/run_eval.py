import glob
import json
import os
from pathlib import Path

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


def setup_dataset():
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
    return dataset.setup("train"), dataset.train_list, dataset.val_list


def setup_model(device):
    model = VideoMAEForPreTraining.from_pretrained("standardmodelbio/smb-vision-base", trust_remote_code=True).to(
        device
    )
    return model


def generate_embedding(model, image, device):
    image = image.to(device)
    with torch.no_grad():
        embedding = model.videomae(image.unsqueeze(0))
    return embedding


def save_embedding(embedding, save_path):
    tensors = {"embedding": embedding.last_hidden_state}
    save_file(tensors, save_path)


def process_split(data, file_list, split_name):
    print(f"\nProcessing {split_name} split...")
    for i, item in enumerate(data[split_name]):
        image = item["image"]
        print(f"Processing image {i + 1} with shape: {image.shape}")

        filepath = Path(file_list[i]["image"])
        save_name = filepath.stem.replace(".nii", "")
        save_path = Path("embeddings") / f"{save_name}.safetensors"

        embedding = generate_embedding(model, image, device)
        save_embedding(embedding, save_path)


if __name__ == "__main__":
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup dataset and model
    data, train_list, val_list = setup_dataset()
    model = setup_model(device)

    # Create output directory
    os.makedirs("embeddings", exist_ok=True)

    # Process train and validation splits
    process_split(data, train_list, "train")
    process_split(data, val_list, "validation")
