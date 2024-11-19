import json
import os


def create_dataset_json(data_dir, output_file="dataset.json", val_split=0.2):
    files = []
    for root, dirs, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename.endswith(".nii.gz"):
                file_path = os.path.join(root, filename)
                files.append({"image": file_path})

    # Calculate split indices
    total = len(files)
    val_size = int(total * val_split)
    train_size = total - val_size
    print(total, train_size, val_size)

    # Create dataset dict
    dataset = {"train": files[:train_size], "validation": files[train_size:]}

    # Write to json file
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=4)

    return dataset


if __name__ == "__main__":
    create_dataset_json(
        "../nifti", output_file="../data/lung-ct-4k-mim.json", val_split=0.01
    )
