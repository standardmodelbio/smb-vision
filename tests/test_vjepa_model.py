import os
import sys
from pathlib import Path


# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import pytest
import torch
from loguru import logger

from src.models.vjepa.configuration_vjepa import VJEPA2Config
from src.models.vjepa.modeling_vjepa import VJEPA2Model


def create_dummy_config():
    """Create a lightweight dummy configuration for testing."""
    config = VJEPA2Config(
        hidden_size=64,  # Small hidden size for testing
        num_hidden_layers=2,  # Few layers for testing
        num_attention_heads=4,  # Few attention heads
        intermediate_size=128,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=32,  # Small image size for testing
        patch_size=8,
        in_chans=1,
        qkv_bias=True,
        drop_path_rate=0.0,
        # Predictor specific configs
        pred_hidden_size=64,
        pred_num_hidden_layers=2,
        pred_num_attention_heads=4,
        pred_mlp_ratio=4.0,
        pred_zero_init_mask_tokens=True,
        pred_num_mask_tokens=1,
        # Video specific configs
        frames_per_clip=4,  # Small number of frames
        tubelet_size=2,
        crop_size=32,
    )
    return config


def create_dummy_input(batch_size=2):
    """Create dummy input video tensor."""
    # Shape: [batch_size, num_frames, channels, height, width]
    return torch.randn(batch_size, 4, 1, 32, 32)


def test_vjepa_model_forward():
    """Test the forward pass of the VJEPA model."""
    # Create dummy config and model
    config = create_dummy_config()
    model = VJEPA2Model(config)

    # Create dummy input
    dummy_input = create_dummy_input()

    # Test forward pass
    outputs = model(pixel_values_videos=dummy_input)

    # Check output shapes
    assert outputs.last_hidden_state.shape == (2, 32, 64)  # batch_size, num_patches, hidden_size
    assert outputs.masked_hidden_state is not None
    assert outputs.predictor_output is not None

    # Check predictor output shapes
    pred_output = outputs.predictor_output
    assert pred_output.last_hidden_state.shape == (2, 32, 64)  # batch_size, num_patches, hidden_size
    assert pred_output.target_hidden_state.shape == (2, 32, 64)


def test_vjepa_model_with_masks(caplog):
    """Test the VJEPA model with custom context and target masks."""
    # Set loguru to capture output
    logger.remove()  # Remove default handler
    logger.add(sys.stderr, level="INFO")  # Add stderr handler

    config = create_dummy_config()
    model = VJEPA2Model(config)

    # Create dummy input
    dummy_input = create_dummy_input()

    # Create random masks
    batch_size = 2
    num_patches = 32

    # Create random permutations for each batch
    context_mask = [
        torch.randperm(num_patches, device=dummy_input.device)[:20].unsqueeze(0).repeat(batch_size, 1),
        torch.randperm(num_patches, device=dummy_input.device)[:20].unsqueeze(0).repeat(batch_size, 1),
    ]
    target_mask = [
        torch.randperm(num_patches, device=dummy_input.device)[20:].unsqueeze(0).repeat(batch_size, 1),
        torch.randperm(num_patches, device=dummy_input.device)[20:].unsqueeze(0).repeat(batch_size, 1),
    ]

    # Log mask information
    logger.info("\nContext Masks:")
    for i, mask in enumerate(context_mask):
        logger.info(f"Context Mask {i}:")
        logger.info(f"Shape: {mask.shape}")
        logger.info(f"Values: {mask[0].tolist()}")  # Print first batch as example

    logger.info("\nTarget Masks:")
    for i, mask in enumerate(target_mask):
        logger.info(f"Target Mask {i}:")
        logger.info(f"Shape: {mask.shape}")
        logger.info(f"Values: {mask[0].tolist()}")  # Print first batch as example

    # Test forward pass with masks
    outputs = model(pixel_values_videos=dummy_input, context_mask=context_mask, target_mask=target_mask)

    # Check output shapes
    assert outputs.last_hidden_state.shape == (2, 32, 64)
    assert outputs.masked_hidden_state.shape == (4, 20, 64)
    assert outputs.predictor_output.last_hidden_state.shape == (4, 12, 64)
    assert outputs.target_hidden_state.shape == outputs.predictor_output.last_hidden_state.shape


def test_vjepa_model_skip_predictor():
    """Test the VJEPA model with predictor skipped."""
    config = create_dummy_config()
    model = VJEPA2Model(config)

    # Create dummy input
    dummy_input = create_dummy_input()

    # Test forward pass with predictor skipped
    outputs = model(pixel_values_videos=dummy_input, skip_predictor=True)

    # Check output shapes
    assert outputs.last_hidden_state.shape == (2, 32, 64)
    assert outputs.predictor_output is None


if __name__ == "__main__":
    pytest.main([__file__])
