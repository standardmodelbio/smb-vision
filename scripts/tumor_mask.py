import numpy as np
from scipy import ndimage


def extract_tumor_and_peritumoral(mask_volume, peritumoral_margin=2, patch_size=(16, 16, 16)):
    """
    Extract tumor and peritumoral regions from a 3D annotation mask.
    Flattens dilated mask into sequence of patches and creates position mask.

    Parameters:
    mask_volume: 3D numpy array (z, y, x) with tumor annotations (1 for tumor, 0 for background)
    peritumoral_margin: Integer specifying the margin size (in voxels) for peritumoral region
    patch_size: Tuple (z,y,x) specifying size of patches to use

    Returns:
    tumor_coords: List of coordinates (z, y, x) for tumor region
    peritumoral_coords: List of coordinates (z, y, x) for peritumoral region
    patch_mask: Binary mask indicating if patches contain tumor (1) or not (0)
    """

    # Get tumor coordinates
    tumor_coords = np.where(mask_volume == 1)
    tumor_coords = list(zip(tumor_coords[0], tumor_coords[1], tumor_coords[2]))

    # Create dilated mask for peritumoral region
    dilated_mask = ndimage.binary_dilation(
        mask_volume,
        structure=np.ones((peritumoral_margin * 2 + 1, peritumoral_margin * 2 + 1, peritumoral_margin * 2 + 1)),
    )

    # Create patch position mask
    z_steps = mask_volume.shape[0] // patch_size[0]
    y_steps = mask_volume.shape[1] // patch_size[1]
    x_steps = mask_volume.shape[2] // patch_size[2]

    patch_mask = np.zeros((z_steps, y_steps, x_steps))

    for z in range(z_steps):
        for y in range(y_steps):
            for x in range(x_steps):
                patch = dilated_mask[
                    z * patch_size[0] : (z + 1) * patch_size[0],
                    y * patch_size[1] : (y + 1) * patch_size[1],
                    x * patch_size[2] : (x + 1) * patch_size[2],
                ]
                if np.any(patch):
                    patch_mask[z, y, x] = 1

    return tumor_coords, patch_mask.flatten()


# Example usage
def main():
    # Create sample data for testing
    volume_shape = (96, 96, 96)
    mask_volume = np.zeros(volume_shape)

    # Create a synthetic tumor mask in the middle
    mask_volume[40:60, 40:60, 40:60] = 1

    # Test parameters
    patch_size = (16, 16, 16)
    peritumoral_margin = 5

    # Call function and get results
    tumor_coords, patch_mask = extract_tumor_and_peritumoral(
        mask_volume, peritumoral_margin=peritumoral_margin, patch_size=patch_size
    )

    # Print test results
    print(f"Volume shape: {volume_shape}")
    print(f"Tumor volume: {len(tumor_coords)}")
    print(f"Number of total patches: {len(patch_mask)}")
    print(f"Number of patches containing tumor/peritumoral region: {np.sum(patch_mask)}")

    # Validate results
    assert len(tumor_coords) > 0, "No tumor coordinates found"
    assert len(patch_mask) == np.prod(np.array(volume_shape) // np.array(patch_size)), "Incorrect patch mask size"

    return tumor_coords, patch_mask


if __name__ == "__main__":
    tumor_coords, patch_mask = main()
