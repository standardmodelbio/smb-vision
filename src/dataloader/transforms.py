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


class PermuteImage(Transform):
    """Permute the dimensions of the image"""

    def __call__(self, data):
        data["image"] = data["image"].permute(3, 0, 1, 2)  # Adjust permutation order as needed
        return data


ct_transforms = {
    "smb-vision": Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=300, b_min=0.0, b_max=1.0, clip=True),
            SpatialPadd(keys=["image"], spatial_size=[16, 16, 16]),
            CenterSpatialCropd(
                roi_size=[512, 512, 320],
                keys=["image"],
            ),
            ToTensord(keys=["image"]),
            PermuteImage(),
        ]
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
        ]
    ),
}
