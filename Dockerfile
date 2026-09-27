# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies & Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    wget \
    aria2 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.4
RUN pip install --no-cache-dir \
    torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Set working directory
WORKDIR /app

# Clone ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app/ComfyUI

WORKDIR /app/ComfyUI

# Install ComfyUI requirements
RUN pip install --no-cache-dir -r requirements.txt

# Clone Custom Nodes
RUN cd custom_nodes && \
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git && \
    git clone https://github.com/kijai/ComfyUI-KJNodes.git

# Install Custom Nodes Dependencies
RUN pip install --no-cache-dir \
    color-matcher \
    mss \
    opencv-python-headless \
    GitPython

# Create model and workflow directories
RUN mkdir -p \
    models/text_encoders \
    models/clip \
    models/diffusion_models \
    models/unet \
    models/vae \
    models/loras \
    user/default/workflows

# Copy pre-configured Tier 2 workflow into ComfyUI default workflows
COPY face_swap_3ref_masked_workflow.json /app/ComfyUI/user/default/workflows/face_swap_3ref_masked_workflow.json

# Copy entrypoint script
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose ComfyUI Web GUI port
EXPOSE 8188

# Launch via entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
