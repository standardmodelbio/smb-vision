import json
from typing import Optional, Sequence

import numpy as np
import torch
import torch.distributed as ptdist
from monai.data import CacheDataset, PersistentDataset, partition_dataset
from monai.data.utils import pad_list_data_collate
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandSpatialCropSamplesd,
    CenterSpatialCropd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
    Transform,
)


class MaskGenerator:
    """
    A class to generate boolean masks for the pretraining task.

    A mask is a 1D tensor of shape (model_patch_size**3,) where the value is either 0 or 1,
    where 1 indicates "masked".
    """

    def __init__(
        self,
        input_size=224,
        depth=96,
        mask_patch_size=32,
        model_patch_size=16,
        mask_ratio=0.6,
    ):
        self.input_size = input_size
        self.depth = depth
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        if self.input_size % self.mask_patch_size != 0:
            raise ValueError("Input size must be divisible by mask patch size")
        if self.depth % self.mask_patch_size != 0:
            raise ValueError("Depth must be divisible by mask patch size")
        if self.mask_patch_size % self.model_patch_size != 0:
            raise ValueError("Mask patch size must be divisible by model patch size")

        self.rand_size = self.input_size // self.mask_patch_size
        self.rand_depth = self.depth // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size**2 * self.rand_depth
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[: self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_depth, self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1).repeat(self.scale, axis=2)

        return torch.tensor(mask.flatten()).bool()


class GenerateMask(Transform):
    def __init__(
        self,
        input_size=224,
        depth=96,
        mask_patch_size=32,
        model_patch_size=16,
        mask_ratio=0.75,
    ):
        self.mask_generator = MaskGenerator(input_size, depth, mask_patch_size, model_patch_size, mask_ratio)

    def __call__(self, inputs):
        inputs["mask"] = self.mask_generator()
        return inputs


class PermuteImage(Transform):
    """Permute the dimensions of the image"""

    def __call__(self, data):
        data["image"] = data["image"].permute(3, 0, 1, 2)  # Adjust permutation order as needed
        return data


class MIMDataset:
    def __init__(
        self,
        json_path: str,
        img_size: int,
        depth: int,
        mask_patch_size: int,
        patch_size: int,
        downsample_ratio: Sequence[float],
        cache_dir: str,
        batch_size: int = 1,
        val_batch_size: int = 1,
        num_workers: int = 4,
        cache_num: int = 0,
        cache_rate: float = 0.0,
        dist: bool = False,
        mask_ratio: float = 0.75,
    ):
        super().__init__()
        self.json_path = json_path
        self.img_size = img_size
        self.depth = depth
        self.mask_patch_size = mask_patch_size
        self.patch_size = patch_size
        self.cache_dir = cache_dir
        self.downsample_ratio = downsample_ratio
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.cache_num = cache_num
        self.cache_rate = cache_rate
        self.dist = dist
        self.mask_ratio = mask_ratio

        data_list = json.load(open(json_path, "r"))

        if "train" in data_list.keys():
            self.train_list = data_list["train"]
        if "validation" in data_list.keys():
            self.val_list = data_list["validation"]

    def val_transforms(
        self,
    ):
        return self.train_transforms()

    def train_transforms(
        self,
    ):
        transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(
                    keys=["image"],
                    pixdim=(1.5, 1.5, 3.0),
                    mode=("bilinear"),
                ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-1000,
                    a_max=1000,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                # CropForegroundd(keys=["image"], source_key="image", allow_smaller=False),
                SpatialPadd(
                    keys=["image"],
                    spatial_size=(self.img_size, self.img_size, self.depth),
                ),
                CenterSpatialCropd(
                    keys=["image"],
                    roi_size=(self.img_size, self.img_size, self.depth),
                ),
                # RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                # RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                # ToTensord(keys=["image"], track_meta=False),
                PermuteImage(),
                # Add custom transform to generate mask
                GenerateMask(
                    input_size=self.img_size,
                    depth=self.depth,
                    mask_patch_size=self.mask_patch_size,
                    model_patch_size=self.patch_size,
                    mask_ratio=self.mask_ratio,
                ),
            ]
        )

        return transforms

    def setup(self, stage: Optional[str] = None):
        # Assign Train split(s) for use in Dataloaders
        if stage in [None, "train"]:
            if self.dist:
                train_partition = partition_dataset(
                    data=self.train_list,
                    num_partitions=ptdist.get_world_size(),
                    shuffle=True,
                    even_divisible=True,
                    drop_last=False,
                )[ptdist.get_rank()]
                valid_partition = partition_dataset(
                    data=self.val_list,
                    num_partitions=ptdist.get_world_size(),
                    shuffle=False,
                    even_divisible=True,
                    drop_last=False,
                )[ptdist.get_rank()]
                # self.cache_num //= ptdist.get_world_size()
            else:
                train_partition = self.train_list
                valid_partition = self.val_list

            if any([self.cache_num, self.cache_rate]) > 0:
                train_ds = CacheDataset(
                    train_partition,
                    cache_num=self.cache_num,
                    cache_rate=self.cache_rate,
                    num_workers=self.num_workers,
                    transform=self.train_transforms(),
                )
                valid_ds = CacheDataset(
                    valid_partition,
                    cache_num=self.cache_num // 4,
                    cache_rate=self.cache_rate,
                    num_workers=self.num_workers,
                    transform=self.val_transforms(),
                )
            else:
                train_ds = PersistentDataset(
                    train_partition,
                    transform=self.train_transforms(),
                    cache_dir=self.cache_dir,
                )
                valid_ds = PersistentDataset(
                    valid_partition,
                    transform=self.val_transforms(),
                    cache_dir=self.cache_dir,
                )

            return {"train": train_ds, "validation": valid_ds}

        if stage in [None, "test"]:
            if any([self.cache_num, self.cache_rate]) > 0:
                test_ds = CacheDataset(
                    self.val_list,
                    cache_num=self.cache_num // 4,
                    cache_rate=self.cache_rate,
                    num_workers=self.num_workers,
                    transform=self.val_transforms(),
                )
            else:
                test_ds = PersistentDataset(
                    self.val_list,
                    transform=self.val_transforms(),
                    cache_dir=self.cache_dir,
                )

            return {"test": test_ds}

        return {"train": None, "validation": None}

    def train_dataloader(self, train_ds):
        # def collate_fn(examples):
        #     pixel_values = torch.stack([example["image"] for example in examples])
        #     mask = torch.stack([example["mask"] for example in examples])
        #     return {"pixel_values": pixel_values, "bool_masked_pos": mask}

        return torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            collate_fn=pad_list_data_collate,
            # collate_fn=collate_fn
            # drop_last=False,
            # prefetch_factor=4,
        )

    def val_dataloader(self, valid_ds):
        return torch.utils.data.DataLoader(
            valid_ds,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            # drop_last=False,
            collate_fn=pad_list_data_collate,
            # prefetch_factor=4,
        )

    def test_dataloader(self, test_ds):
        return torch.utils.data.DataLoader(
            test_ds,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            # drop_last=False,
            collate_fn=pad_list_data_collate,
            # prefetch_factor=4,
        )


if __name__ == "__main__":
    # Initialize dataset with example parameters
    dataset = MIMDataset(
        json_path="/home/user/smb_vision_data.json",
        img_size=224,
        depth=160,
        mask_patch_size=16,
        patch_size=16,
        downsample_ratio=(1.5, 1.5, 3.0),
        cache_dir="/home/user/cache",
        batch_size=1,
        val_batch_size=1,
        num_workers=8,
        cache_num=0,
        cache_rate=0.0,
        dist=False,
        mask_ratio=0.5,
    )

    # Setup dataset and get data loaders
    datasets = dataset.setup("train")
    train_loader = dataset.train_dataloader(datasets["train"])
    val_loader = dataset.val_dataloader(datasets["validation"])

    # Initialize lists to store valid files
    valid_train_files = []
    valid_val_files = []

    # Process training data
    print("Processing training data...")
    for i, data in enumerate(train_loader):
        try:
            # Check if data can be loaded correctly
            print(data["image"].shape)
            print(data["mask"].shape)
            # If no error, add file to valid list
            valid_train_files.append(dataset.train_list[i])
            print(dataset.train_list[i])
            if i % 100 == 0:
                print(f"Processed {i} training files")
        except Exception as e:
            print(f"Error in training file {i}: {str(e)}")
            continue

    # Process validation data
    print("\nProcessing validation data...")
    for i, data in enumerate(val_loader):
        try:
            # Check if data can be loaded correctly
            _ = data["image"].shape
            _ = data["mask"].shape
            # If no error, add file to valid list
            valid_val_files.append(dataset.val_list[i])
            if i % 100 == 0:
                print(f"Processed {i} validation files")
        except Exception as e:
            print(f"Error in validation file {i}: {str(e)}")
            continue

    # Save valid files to json
    valid_files = {"train": valid_train_files, "validation": valid_val_files}

    with open("valid_files.json", "w") as f:
        json.dump(valid_files, f, indent=4)

    print("\nProcessing complete!")
    print(f"Valid training files: {len(valid_train_files)}/{len(dataset.train_list)}")
    print(f"Valid validation files: {len(valid_val_files)}/{len(dataset.val_list)}")
