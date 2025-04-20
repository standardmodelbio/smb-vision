from typing import Optional, Sequence

import torch
import torch.distributed as ptdist
from monai.data import (
    CacheDataset,
    Dataset,
    partition_dataset,
)
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    SpatialPadd,
    ToTensord,
    Transform,
)


class PermuteImage(Transform):
    """Permute the dimensions of the image"""

    def __call__(self, data):
        data["image"] = data["image"].permute(3, 0, 1, 2)  # Adjust permutation order as needed
        return data


class CTDataset:
    def __init__(
        self,
        data_list,
        img_size: int,
        depth: int,
        downsample_ratio: Optional[Sequence[float]] = None,
        batch_size: int = 1,
        val_batch_size: int = 1,
        num_workers: int = 4,
        cache_num: int = 0,
        cache_rate: float = 0.0,
        cache_dir: Optional[str] = None,
        dist: bool = False,
        bf16: bool = False,
    ):
        super().__init__()
        self.data_list = data_list
        self.img_size = img_size
        self.depth = depth
        self.cache_dir = cache_dir
        self.downsample_ratio = downsample_ratio
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.cache_num = cache_num
        self.cache_rate = cache_rate
        self.dist = dist
        self.bf16 = bf16

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
                # Spacingd(
                #     keys=["image"],
                #     pixdim=self.downsample_ratio,
                #     mode=("bilinear"),
                # ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-1000,
                    a_max=300,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                # CropForegroundd(keys=["image"], source_key="image"),
                CenterSpatialCropd(keys=["image"], roi_size=(self.img_size, self.img_size, self.depth)),
                SpatialPadd(
                    keys=["image"],
                    spatial_size=(16, 16, 16),
                ),
                ToTensord(keys=["image"], dtype=torch.bfloat16 if self.bf16 else torch.float32),
                PermuteImage(),
            ]
        )

        return transforms

    def setup(
        self,
    ):
        if self.dist:
            train_partition = partition_dataset(
                data=self.data_list,
                num_partitions=ptdist.get_world_size(),
                shuffle=True,
                even_divisible=True,
                drop_last=False,
            )[ptdist.get_rank()]
        else:
            train_partition = self.data_list

        if any([self.cache_num, self.cache_rate]) > 0:
            train_ds = CacheDataset(
                train_partition,
                cache_num=self.cache_num,
                cache_rate=self.cache_rate,
                num_workers=self.num_workers,
                transform=self.train_transforms(),
            )
        else:
            train_ds = Dataset(
                train_partition,
                transform=self.train_transforms(),
            )

        return train_ds
