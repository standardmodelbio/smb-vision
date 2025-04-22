from typing import Optional

import torch.distributed as ptdist
from monai.data import (
    CacheDataset,
    Dataset,
    partition_dataset,
)

from .transforms import ct_transforms


class CTDataset:
    def __init__(
        self,
        data_list,
        args,
    ):
        super().__init__()
        self.data_list = data_list
        self.num_workers = args.num_workers
        self.cache_num = args.cache_num
        self.cache_rate = args.cache_rate
        self.cache_dir = args.cache_dir
        self.dist = args.dist
        self.model_class = args.model_class

    def val_transforms(
        self,
        model_class: str,
    ):
        return ct_transforms[model_class]

    def train_transforms(
        self,
        model_class: str,
    ):
        return ct_transforms[model_class]

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
                transform=self.train_transforms(self.model_class),
            )
        else:
            train_ds = Dataset(
                train_partition,
                transform=self.train_transforms(self.model_class),
            )

        return train_ds
