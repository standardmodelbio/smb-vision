# SMB Vision

A deep learning project for processing and analyzing 3D medical images using masked image modeling and embedding generation.

## Overview

This project provides tools for:
1. Pre-training vision transformers on 3D medical images using masked image modeling (MIM)
2. Generating embeddings from medical images using pre-trained models
3. Fine-tuning pretrained MIM-3D model on downstream classification/regression tasks

## Installation

```bash
# Install UV for faster python env build up
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.loca/bin/env
export UV_LINK_MODE=copy

# Clone SMB-Vision package
git clone https://github.com/standardmodelbio/smb-vision.git
cd smb-vision

# Build virtual env for the package
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .
uv pip install flash-attn --no-build-isolation
```

## Project Structure

```
smb-vision/
├── src/
│   ├── dataloader/
│   │   ├── mim.py        # Dataset classes for masked image modeling
│   │   └── load.py       # Dataset classes for inference
│   ├── run_mim.py        # Script for pre-training using MIM
│   ├── run_classification.py  # Script for fine-tuning classification/regression
│   └── run_inference.py  # Script for generating embeddings
├── scripts/
│   ├── build_train_file.py  # Script to prepare training data
│   └── training/
│       ├── run_mim.sh    # Shell script for MIM training
│       └── run_cls.sh    # Shell script for classification training
```

## Usage

### 1. Pre-training with Masked Image Modeling

```bash
# Using shell script
./scripts/training/run_mim.sh

# Or using Python script
python src/run_mim.py \
    --json_path path/to/dataset.json \
    --output_dir ./outputs \
    --image_size 512 \
    --depth 320 \
    --patch_size 16 \
    --mask_patch_size 32 \
    --mask_ratio 0.65 \
    --learning_rate 1e-4 \
    --num_train_epochs 100 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1
```

### 2. Fine-tuning for Classification/Regression

```bash
# Using shell script
./scripts/training/run_cls.sh

# Or using Python script
python src/run_classification.py \
    --json_path path/to/dataset.json \
    --model_name_or_path standardmodelbio/smb-vision-base \
    --task_type classification \
    --num_labels 2 \
    --output_dir ./outputs \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4
```

### 3. Generating Embeddings

```bash
python src/run_inference.py \
    --json_path path/to/dataset.json \
    --model_name standardmodelbio/smb-vision-base-20250122 \
    --output_dir ./embeddings \
    --img_size 512 \
    --depth 320
```

## Data Format

### For Pre-training and Classification

The expected format for each file type:

1. **CSV/Parquet**

| image_path | label | split |
|------------|-------|-------|
| path/to/image1.nii.gz | 1 | train |
| path/to/image2.nii.gz | 0 | train |
| path/to/image3.nii.gz | 1 | val |

2. **JSON**

```json
{
  "train": [
    {"image": "path/to/image1.nii.gz", "label": 0},
    {"image": "path/to/image2.nii.gz", "label": 1},
    ...
  ],
  "validation": [
    {"image": "path/to/image3.nii.gz", "label": 0},
    {"image": "path/to/image4.nii.gz", "label": 1},
    ...
  ],
  "test": [...]
}
```
or
```json
[
  {"image": "path/to/image1.nii.gz", "label": 0},
  {"image": "path/to/image2.nii.gz", "label": 1},
    ...
]
```


### For Inference

```json
[
  {"image": "path/to/image1.nii.gz"},
  {"image": "path/to/image2.nii.gz"},
  {"image": "path/to/image3.nii.gz"},
  {"image": "path/to/image4.nii.gz"},
  ...
]
```

The generated embeddings will be saved in the specified output directory with the following structure:

```
embeddings/
├── image1.npy      # Numpy array for image 1
├── image2.npy      # Numpy array for image 2
├── image3.npy      # Numpy array for image 3
├── image4.npy      # Numpy array for image 4
└── metadata.json   # Mapping between files and embeddings
```

## Features

- Support for 3D medical images (NIFTI format)
- Masked image modeling pre-training
- Fine-tuning for classification and regression tasks
- Embedding generation using pre-trained models
- Configurable image size, depth, and patch sizes
- Mixed precision training (bf16)
- Gradient checkpointing for memory efficiency
- Distributed training support
- Weights & Biases integration for experiment tracking
- Data caching for improved performance

## Requirements

- Python 3.8+
- PyTorch
- transformers
- MONAI
- safetensors
- numpy
- accelerate
- wandb

## Acknowledgments

This project uses the following open-source projects:
- 🤗 Transformers
- MONAI
- PyTorch
- Weights & Biases

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
