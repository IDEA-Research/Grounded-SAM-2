FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# Arguments to build Docker Image using CUDA
ARG USE_CUDA=0
ARG TORCH_ARCH="7.0;7.5;8.0;8.6"

ENV AM_I_DOCKER=True
ENV BUILD_WITH_CUDA="${USE_CUDA}"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_ARCH}"
ENV CUDA_HOME=/usr/local/cuda-12.1/
# Ensure CUDA is correctly set up
ENV PATH=/usr/local/cuda-12.1/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH}

# Install required packages and specific gcc/g++
RUN apt-get update && apt-get install --no-install-recommends wget ffmpeg=7:* \
    libsm6=2:* libxext6=2:* git=1:* nano vim=2:* ninja-build gcc-10 g++-10 -y \
    && apt-get clean && apt-get autoremove && rm -rf /var/lib/apt/lists/*

ENV CC=gcc-10
ENV CXX=g++-10

RUN mkdir -p /home/appuser/Grounded-SAM-2
COPY . /home/appuser/Grounded-SAM-2/

WORKDIR /home/appuser/Grounded-SAM-2


# Install essential Python packages
RUN python -m pip install --upgrade pip setuptools wheel numpy \
    opencv-python transformers supervision pycocotools addict yapf timm

# Install segment_anything package in editable mode
RUN python -m pip install -e .

# Install grounding dino 
RUN python -m pip install --no-build-isolation -e grounding_dino
