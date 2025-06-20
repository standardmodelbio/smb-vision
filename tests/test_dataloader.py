import os
import random
import sys
from pathlib import Path


# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)


import torch
from torch.utils.data import DataLoader

from run_vjepa import collate_fn
from src.dataloader.load import CTPersistentDataset
from src.dataloader.transforms import ct_transforms


def test_ct_persistent_dataloader():
    """Test the CTPersistentDataset with the dummy dataset."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    # Get the absolute path to the dummy dataset
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = "/home/ec2-user/smb-vision/dummy_data/dummy_dataset.json"

    # Create train dataset
    train_dataset = CTPersistentDataset(
        data_path=data_path,
        split="train",
        transform=ct_transforms["vjepa"],
    )

    # Create validation dataset
    val_dataset = CTPersistentDataset(
        data_path=data_path,
        split="validation",
        transform=ct_transforms["vjepa"],
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Test train dataloader
    print("\nTesting train dataloader:")
    for batch_idx, batch in enumerate(train_dataloader):
        print(f"\nTrain Batch {batch_idx + 1}:")
        print(f"Keys in batch: {batch.keys()}")

        # Check shapes
        print(f"pixel_values shape: {batch['pixel_values'].shape}")
        print(f"context_mask shape: {batch['context_mask'][0].shape}")
        print(f"target_mask shape: {batch['target_mask'][0].shape}")

        # Check data types
        print(f"pixel_values dtype: {batch['pixel_values'].dtype}")
        print(f"context_mask dtype: {batch['context_mask'][0].dtype}")
        print(f"target_mask dtype: {batch['target_mask'][0].dtype}")

        # Check value ranges
        print(f"pixel_values range: [{batch['pixel_values'].min():.3f}, {batch['pixel_values'].max():.3f}]")
        print(f"context_mask unique values: {torch.unique(batch['context_mask'][0])}")
        print(f"target_mask unique values: {torch.unique(batch['target_mask'][0])}")

        # Only test first batch for brevity
        if batch_idx == 0:
            break

    # Test validation dataloader
    print("\nTesting validation dataloader:")
    for batch_idx, batch in enumerate(val_dataloader):
        print(f"\nValidation Batch {batch_idx + 1}:")
        print(f"Keys in batch: {batch.keys()}")

        # Check shapes
        print(f"pixel_values shape: {batch['pixel_values'].shape}")
        print(f"context_mask shape: {batch['context_mask'][0].shape}")
        print(f"target_mask shape: {batch['target_mask'][0].shape}")

        # Check data types
        print(f"pixel_values dtype: {batch['pixel_values'].dtype}")
        print(f"context_mask dtype: {batch['context_mask'][0].dtype}")
        print(f"target_mask dtype: {batch['target_mask'][0].dtype}")

        # Check value ranges
        print(f"pixel_values range: [{batch['pixel_values'].min():.3f}, {batch['pixel_values'].max():.3f}]")
        print(f"context_mask unique values: {torch.unique(batch['context_mask'][0])}")
        print(f"target_mask unique values: {torch.unique(batch['target_mask'][0])}")

        # Only test first batch for brevity
        if batch_idx == 0:
            break


def test_dataset_info():
    """Test dataset information and properties."""
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = "/home/ec2-user/smb-vision/dummy_data/dummy_dataset.json"

    # Create train dataset
    train_dataset = CTPersistentDataset(
        data_path=data_path,
        split="train",
        transform=ct_transforms["vjepa"],
    )

    print("\nDataset Information:")
    print(f"Number of training samples: {len(train_dataset)}")

    # Test single sample
    sample = train_dataset[0]
    print("\nSingle sample information:")
    print(f"Keys in sample: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Context mask shape: {sample['context_mask'].shape}")
    print(f"Target mask shape: {sample['target_mask'].shape}")


if __name__ == "__main__":
    print("Testing CTPersistentDataset with dummy data...")
    test_dataset_info()
    test_ct_persistent_dataloader()
