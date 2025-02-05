# SMB Vision

A deep learning project for processing and analyzing 3D medical images using masked image modeling and embedding generation.

## Overview

This project provides tools for:
1. Pre-training vision transformers on 3D medical images using masked image modeling (MIM)
2. Generating embeddings from medical images using pre-trained models

## Installation

```bash
conda create -n vision python=3.11
conda activate vision

git clone https://github.com/standardmodelbio/smb-vision.git
cd smb-vision
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Project Structure

```
smb-vision/
├── src/
│   ├── dataloader/
│   │   ├── mim.py        # Dataset classes for masked image modeling
│   │   └── load.py       # Dataset classes for inference
│   ├── run_mim.py        # Script for pre-training using MIM
│   └── run_inference.py  # Script for generating embeddings
```

## Usage

### 1. Pre-training with Masked Image Modeling

```bash
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

### 2. Generating Embeddings

```bash
python src/run_inference.py \
    --json_path path/to/dataset.json \
    --model_name standardmodelbio/smb-vision-base-20250122 \
    --output_dir ./embeddings \
    --img_size 512 \
    --depth 320
```

## Data Format

The dataset should be provided as a JSON file with the following structure:

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
└── metadata.json        # Mapping between files and embeddings
```

## Features

- Support for 3D medical images (NIFTI format)
- Masked image modeling pre-training
- Embedding generation using pre-trained models
- Configurable image size, depth, and patch sizes
- CUDA support for GPU acceleration
- Logging and error handling
- Data caching for improved performance

## Requirements

- Python 3.8+
- PyTorch
- transformers
- MONAI
- safetensors
- numpy

## Acknowledgments

This project uses the following open-source projects:
- 🤗 Transformers
- MONAI
- PyTorch

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
