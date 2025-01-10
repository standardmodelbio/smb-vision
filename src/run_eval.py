import glob
import json
import os

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
    train_dir = "../mdanderson/public_data/train/CT/"
    val_dir = "../mdanderson/public_data/valid/CT/"
    json_path = "../data/dataset.json"
    build_json_from_nifti_files(train_dir, val_dir, json_path)

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
    data = dataset.setup("fit")

    for batch in dataset.train_dataloader(data["train"]):
        image = batch["image"]
        break

    # model = VideoMAEForPreTraining.from_pretrained(
    #     "standardmodelbio/smb-vision-base",
    #     trust_remote_code=True,
    # )

    # embedding = model.videomae(image)
    # print(embedding.shape)
