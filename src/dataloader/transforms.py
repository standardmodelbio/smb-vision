import math

import numpy as np
import torch
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
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
        mask = self.mask_generator()
        inputs["mask"] = mask

        return inputs


class PermuteImage(MapTransform):
    """Permute the dimensions of the image"""

    def __init__(self, keys=["image"], allow_missing_keys=False):
        MapTransform.__init__(self, keys, allow_missing_keys)

    def __call__(self, data):
        data["image"] = data["image"].permute(3, 0, 1, 2)  # Adjust permutation order as needed

        return data


class VJEPAMaskGenerator(Transform):
    """
    Generate context and target masks for V-JEPA training on 3D CT volumes.
    Based on the original Video-JEPA implementation but adapted for 3D spatial data.
    """

    def __init__(
        self,
        input_size=(224, 224, 160),
        patch_size=(16, 16, 16),
        pred_mask_scale=(0.2, 0.8),
        aspect_ratio=(0.3, 3.0),
        num_blocks=1,
        max_keep=None,
        inv_block=False,
        full_complement=False,
        pred_full_complement=False,
    ):
        super().__init__()

        # Convert single values to tuples if needed
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 3
        if not isinstance(patch_size, tuple):
            patch_size = (patch_size,) * 3

        self.input_size = input_size
        self.patch_size = patch_size

        # Calculate dimensions in patch space
        self.depth = input_size[0] // patch_size[0]
        self.height = input_size[1] // patch_size[1]
        self.width = input_size[2] // patch_size[2]

        # Mask generation parameters
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.num_blocks = num_blocks
        self.max_keep = max_keep
        self.inv_block = inv_block
        self.full_complement = full_complement
        self.pred_full_complement = pred_full_complement

    def _sample_block_size(self, generator):
        # Sample block mask scale
        min_s, max_s = self.pred_mask_scale
        mask_scale = min_s + torch.rand(1, generator=generator).item() * (max_s - min_s)
        num_keep = int(self.depth * self.height * self.width * mask_scale)

        # Sample block aspect-ratio
        min_ar, max_ar = self.aspect_ratio
        aspect_ratio = min_ar + torch.rand(1, generator=generator).item() * (max_ar - min_ar)

        # Compute block dimensions
        # For 3D, we use two aspect ratios to determine the shape
        ar1 = aspect_ratio
        ar2 = 1.0 / aspect_ratio

        # Calculate dimensions maintaining the aspect ratios
        d = int(round(math.pow(num_keep * ar1 * ar2, 1 / 3)))
        h = int(round(d * ar1))
        w = int(round(d * ar2))

        # Ensure dimensions don't exceed patch space
        d = min(d, self.depth)
        h = min(h, self.height)
        w = min(w, self.width)

        return (d, h, w)

    def _sample_block_mask(self, b_size):
        d, h, w = b_size
        start_d = torch.randint(0, self.depth - d + 1, (1,))
        start_h = torch.randint(0, self.height - h + 1, (1,))
        start_w = torch.randint(0, self.width - w + 1, (1,))

        mask = torch.ones((self.depth, self.height, self.width), dtype=torch.int32)
        mask[start_d : start_d + d, start_h : start_h + h, start_w : start_w + w] = 0

        return mask

    def __call__(self, data):
        # Generate random seed for reproducibility
        seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator()
        generator.manual_seed(seed)

        # Sample block size
        block_size = self._sample_block_size(generator)

        # Generate masks
        mask_e = torch.ones((self.depth, self.height, self.width), dtype=torch.int32)
        for _ in range(self.num_blocks):
            mask_e *= self._sample_block_mask(block_size)

        # Convert to indices
        mask_e = mask_e.flatten()
        mask_p = torch.argwhere(mask_e == 0).squeeze()
        mask_e = torch.nonzero(mask_e).squeeze()

        # Handle full complement cases
        if self.full_complement:
            total_patches = self.depth * self.height * self.width
            mask_p = torch.tensor(set(range(total_patches)) - set(mask_e.tolist()), dtype=mask_e.dtype)
        elif self.pred_full_complement:
            total_patches = self.depth * self.height * self.width
            mask_e = torch.tensor(set(range(total_patches)) - set(mask_p.tolist()), dtype=mask_p.dtype)

        # Apply max_keep if specified
        if self.max_keep is not None:
            mask_e = mask_e[: self.max_keep]
            mask_p = mask_p[: self.max_keep]

        # Add masks to data
        if self.inv_block:
            data["context_mask"] = mask_p
            data["target_mask"] = mask_e
        else:
            data["context_mask"] = mask_e
            data["target_mask"] = mask_p

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
            # ToTensord(keys=["image"]),
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
    "vjepa": Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.5), mode=("bilinear")),
            ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
            SpatialPadd(keys=["image"], spatial_size=[384, 384, 256]),
            CenterSpatialCropd(
                roi_size=[384, 384, 256],
                keys=["image"],
            ),
            ToTensord(keys=["image"]),
            VJEPAMaskGenerator(
                input_size=(384, 384, 256),
                patch_size=(16, 16, 16),
                pred_mask_scale=(0.2, 0.8),
                aspect_ratio=(0.3, 3.0),
                num_blocks=3,
            ),
            PermuteImage(),
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
            # ToTensord(keys=["image"]),
            PermuteImage(),
        ],
    ),
    "dinov2": Compose(
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
            # ToTensord(keys=["image"]),
            # PermuteImage(),
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
