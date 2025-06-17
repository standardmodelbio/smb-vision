import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dataloader.transforms import VJEPAMaskGenerator


def create_synthetic_volume(shape=(224, 224, 16)):
    """Create a synthetic 3D volume with a simple pattern."""
    x = np.linspace(-1, 1, shape[0])
    y = np.linspace(-1, 1, shape[1])
    z = np.linspace(-1, 1, shape[2])
    X, Y, Z = np.meshgrid(x, y, z)
    volume = np.sin(X * 5) * np.cos(Y * 5) * np.sin(Z * 5)
    return torch.from_numpy(volume).float()


def visualize_masks(volume, context_mask, target_mask, patch_size=(4, 4, 4), save_path=None):
    """Visualize the original volume and masks."""
    # Create a figure with 8 rows (one for each slice) and 3 columns (original, context, target)
    fig, axes = plt.subplots(16, 3, figsize=(15, 60))

    # Calculate patch grid dimensions
    patch_grid = (volume.shape[0] // patch_size[0], volume.shape[1] // patch_size[1], volume.shape[2] // patch_size[2])

    # Create patch-based masks
    context_mask_3d = torch.zeros(patch_grid, dtype=torch.bool)
    target_mask_3d = torch.zeros(patch_grid, dtype=torch.bool)

    # Convert flat indices to 3D patch indices
    context_mask_3d.view(-1)[context_mask] = True
    target_mask_3d.view(-1)[target_mask] = True

    # Expand masks to full volume size
    context_mask_full = (
        context_mask_3d.repeat_interleave(patch_size[0], dim=0)
        .repeat_interleave(patch_size[1], dim=1)
        .repeat_interleave(patch_size[2], dim=2)
    )
    target_mask_full = (
        target_mask_3d.repeat_interleave(patch_size[0], dim=0)
        .repeat_interleave(patch_size[1], dim=1)
        .repeat_interleave(patch_size[2], dim=2)
    )

    # Plot all slices
    for slice_idx in range(volume.shape[2]):
        # Plot original volume
        axes[slice_idx, 0].imshow(volume[:, :, slice_idx], cmap="gray")
        axes[slice_idx, 0].set_title(f"Original Volume (Slice {slice_idx})")

        # Plot context mask
        axes[slice_idx, 1].imshow(context_mask_full[:, :, slice_idx], cmap="gray")
        axes[slice_idx, 1].set_title(f"Context Mask (Slice {slice_idx})")

        # Plot target mask
        axes[slice_idx, 2].imshow(target_mask_full[:, :, slice_idx], cmap="gray")
        axes[slice_idx, 2].set_title(f"Target Mask (Slice {slice_idx})")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def test_vjepa_masking():
    """Test V-JEPA masking functionality."""
    # Create synthetic volume
    volume = create_synthetic_volume()

    # Prepare data dictionary
    data = {"image": volume}

    # Initialize transforms
    mask_generator = VJEPAMaskGenerator(
        input_size=(224, 224, 16),
        patch_size=(16, 16, 16),
        pred_mask_scale=(0.2, 0.8),
        aspect_ratio=(0.3, 3.0),
        num_blocks=3,
    )

    # Apply transforms
    transformed_data = mask_generator(data)

    # Get masks
    context_mask = transformed_data["context_mask"]
    target_mask = transformed_data["target_mask"]

    # Print mask information
    print("Context mask indices:", context_mask.shape)
    print("Target mask indices:", target_mask.shape)
    print("Number of context patches:", len(context_mask))
    print("Number of target patches:", len(target_mask))

    # Verify mask properties
    total_patches = 224 * 224 * 16 // (16 * 16 * 16)  # Total number of patches
    assert len(context_mask) + len(target_mask) == total_patches, "Masks should cover all patches"
    assert len(set(context_mask.tolist()) & set(target_mask.tolist())) == 0, "Masks should be disjoint"

    # Visualize results
    os.makedirs("tests/outputs", exist_ok=True)
    visualize_masks(volume, context_mask, target_mask, patch_size=(16, 16, 16), save_path="tests/outputs/vjepa_masks.png")

    print("Test passed! Check tests/outputs/vjepa_masks.png for visualization.")


def test_multiple_masks():
    """Test V-JEPA masking with different configurations."""
    volume = create_synthetic_volume()
    data = {"image": volume}

    # Test different configurations
    configs = [
        {
            "name": "default",
            "params": {
                "input_size": (224, 224, 16),
                "patch_size": (16, 16, 16),
                "pred_mask_scale": (0.2, 0.8),
                "aspect_ratio": (0.3, 3.0),
                "num_blocks": 1,
            },
        },
        {
            "name": "multiple_blocks",
            "params": {
                "input_size": (224, 224, 16),
                "patch_size": (16, 16, 16),
                "pred_mask_scale": (0.2, 0.8),
                "aspect_ratio": (0.3, 3.0),
                "num_blocks": 3,
            },
        },
        {
            "name": "larger_scale",
            "params": {
                "input_size": (224, 224, 16),
                "patch_size": (16, 16, 16),
                "pred_mask_scale": (0.4, 0.9),
                "aspect_ratio": (0.3, 3.0),
                "num_blocks": 1,
            },
        },
    ]

    for config in configs:
        mask_generator = VJEPAMaskGenerator(**config["params"])
        transformed_data = mask_generator(data.copy())

        # Visualize results
        visualize_masks(
            volume,
            transformed_data["context_mask"],
            transformed_data["target_mask"],
            patch_size=(16, 16, 16),
            save_path=f"tests/outputs/vjepa_masks_{config['name']}.png",
        )

        print(f"Generated masks for {config['name']} configuration")


if __name__ == "__main__":
    test_vjepa_masking()
    test_multiple_masks()
