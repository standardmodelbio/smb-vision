import glob
import json
import os
import random


random.seed(42)


def collect_nifti_files(directory_path, output_file="nifti_files.json", val_size=100):
    # Get all nifti files in directory and subdirectories
    nifti_files = glob.glob(os.path.join(directory_path, "**", "*.nii.gz"), recursive=True)

    # Randomly shuffle files
    random.shuffle(nifti_files)

    # Split into train and val based on validation size
    val_size = min(val_size, len(nifti_files))
    train_files = nifti_files[val_size:]
    val_files = nifti_files[:val_size]

    # Convert to {"image": path} format
    train_records = [{"image": path} for path in train_files]
    val_records = [{"image": path} for path in val_files]

    # Create dictionary
    data_split = {"train": train_records, "validation": val_records}

    # Save to json
    with open(output_file, "w") as f:
        json.dump(data_split, f, indent=2)

    return data_split


def main():
    # Test the function
    directory = "../data/nifti_files/"  # Replace with actual path
    data = collect_nifti_files(directory, "./smb-vision-train-mim.json", val_size=100)
    print(f"Found {len(data['train'])} training files")
    print(f"Found {len(data['validation'])} validation files")


if __name__ == "__main__":
    main()
