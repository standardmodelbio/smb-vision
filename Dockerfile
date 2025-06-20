# Use PyTorch base image with CUDA support
FROM nvcr.io/nvidia/pytorch:25.01-py3

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_ROOT_USER_ACTION=ignore \
    MAX_JOBS=32 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    && apt-get clean

# Change pip source
RUN python -m pip install --upgrade pip

# Uninstall nv-pytorch fork
RUN pip uninstall -y torch torchvision torchaudio \
pytorch-quantization pytorch-triton torch-tensorrt \
transformer-engine flash-attn apex megatron-core \
xgboost opencv grpcio

# Install Python dependencies
# Install torch-2.7.0+cu126 + vllm-0.9.1
RUN pip install --no-cache-dir "torch==2.7.0" "torchvision==0.22.0" "torchaudio==2.7.0" tensordict torchdata \
    "transformers[hf_xet]>=4.51.0" accelerate datasets peft \
    "numpy<2.0.0" "pyarrow>=15.0.0" "grpcio>=1.62.1" "optree>=0.13.0" pandas \
    wandb nibabel monai lifelines loguru ruff

# Install flash-attn-2.8.0.post2
RUN ABI_FLAG=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')") && \
    URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abi${ABI_FLAG}-cp312-cp312-linux_x86_64.whl" && \
    wget -nv -P /opt/tiger "${URL}" && \
    pip install --no-cache-dir "/opt/tiger/$(basename ${URL})"

# Set working directory
WORKDIR /workspace

COPY . /workspace/smb-vision/

# Set entrypoint
ENTRYPOINT ["/bin/bash", "-c", "exec bash"]