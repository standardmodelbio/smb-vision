import numpy as np
import torch
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
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

        return torch.tensor(mask.flatten())


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


ct_transforms = {
    "mim": Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3.0), mode=("bilinear")),
            ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
            SpatialPadd(keys=["image"], spatial_size=[224, 224, 160]),
            CenterSpatialCropd(
                roi_size=[224, 224, 160],
                keys=["image"],
            ),
            ToTensord(keys=["image"]),
            PermuteImage(),
            GenerateMask(
                input_size=224,
                depth=160,
                mask_patch_size=16,
                model_patch_size=16,
                mask_ratio=0.5,
            ),
        ]
    ),
    "smb-vision": Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3.0), mode=("bilinear")),
            ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
            SpatialPadd(keys=["image"], spatial_size=[224, 224, 160]),
            CenterSpatialCropd(
                roi_size=[224, 224, 160],
                keys=["image"],
            ),
            ToTensord(keys=["image"]),
            PermuteImage(),
        ],
    ),
    "merlin": Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3), mode=("bilinear")),
            ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
            SpatialPadd(keys=["image"], spatial_size=[224, 224, 160]),
            CenterSpatialCropd(
                roi_size=[224, 224, 160],
                keys=["image"],
            ),
            ToTensord(keys=["image"]),
        ],
    ),
}
