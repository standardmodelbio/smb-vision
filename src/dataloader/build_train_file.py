import json
import os
from typing import Union

import nibabel as nib

# Import the MaskGenerator and transforms from mim.py
from mim import GenerateMask, PermuteImage
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandSpatialCropSamplesd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
    Transform,
)


def get_transforms(img_size=384, depth=320, mask_patch_size=32, patch_size=16, mask_ratio=0.75):
    """Create transformation pipeline"""
    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(
                keys=["image"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image", allow_smaller=False),
            RandSpatialCropSamplesd(
                keys=["image"],
                roi_size=(img_size, img_size, depth),
                random_size=False,
                num_samples=1,
            ),
            SpatialPadd(
                keys=["image"],
                spatial_size=(img_size, img_size, depth),
            ),
            # ToTensord(keys=["image"]),
            # PermuteImage(),
            # GenerateMask(
            #     input_size=img_size,
            #     depth=depth,
            #     mask_patch_size=mask_patch_size,
            #     model_patch_size=patch_size,
            #     mask_ratio=mask_ratio,
            # ),
        ]
    )


def verify_transforms(file_dict, transforms):
    """Apply transforms to a single file and verify the output"""
    try:
        transformed = transforms(file_dict)

        # Check image shape
        # print(transformed)
        image = transformed[0][0]["image"]
        # mask = transformed[0][0]["mask"]

        print(image)
        print(f"Image shape: {image.shape}")
        # print(f"Mask shape: {mask.shape}")

        return True
    except Exception as e:
        print(f"Transform failed: {str(e)}")
        return False


def create_dataset_json(data_dir, output_file="dataset.json", val_split: Union[int, float] = 0.2, verify=True):
    files = []
    transforms = get_transforms() if verify else None

    for root, dirs, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename.endswith(".nii.gz"):
                file_path = os.path.join(root, filename)
                file_dict = {"image": file_path}

                try:
                    if verify:
                        print(f"\nVerifying transforms for {filename}")
                        if verify_transforms([file_dict], transforms):
                            files.append(file_dict)
                            print(f"Successfully verified {filename}")
                        else:
                            print(f"Failed to verify {filename}")
                    else:
                        # Just check if file can be loaded
                        img = nib.load(file_path)
                        if len(img.get_fdata().shape) == 3:
                            files.append(file_dict)
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")

    # Calculate split indices
    total = len(files)
    if isinstance(val_split, int):
        val_size = val_split
    else:
        val_size = int(total * val_split)
    train_size = total - val_size
    print(f"train_size is {train_size}, val_size is {val_size}")

    # Create dataset dict
    dataset = {"train": files[:train_size], "validation": files[train_size:]}

    # Write to json file
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=4)

    return dataset


if __name__ == "__main__":
    create_dataset_json(
        "../nifti_files",
        output_file="./smb-vision-large-train-mim.json",
        val_split=100,
        verify=True,  # Set to True to verify transforms
    )
