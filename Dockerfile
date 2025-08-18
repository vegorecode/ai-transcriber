FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git ca-certificates wget libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

ENV HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    TORCH_HOME=/root/.cache/torch \
    CT2_USE_EXPERIMENTAL_PACKED_GEMM=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    PIP_NO_CACHE_DIR=1

# 1) pip
RUN python -m pip install --upgrade pip setuptools wheel

# 2) строго выравниваем torch/vision/audio под CUDA 12.4
RUN python -m pip install --no-cache-dir \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu124

# 3) остальное
COPY builder/requirements.txt /builder/requirements.txt
RUN python -m pip install --no-cache-dir -r /builder/requirements.txt

COPY builder /builder
RUN chmod +x /builder/download_models.sh
RUN --mount=type=secret,id=hf_token /builder/download_models.sh || true

COPY src/ /app/
CMD ["python", "-u", "/app/rp_handler.py"]
