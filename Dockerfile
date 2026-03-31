# Target: NVIDIA Jetson Orin (JetPack 6 / L4T R36).
# PyTorch 2.1 with CUDA is pre-installed in this base image — do NOT reinstall
# torch/torchvision from PyPI; the standard wheels target x86 and will not work
# on aarch64/Jetson.
#
# Using dusty-nv's image (Docker Hub, no auth required).
# Official NVIDIA alternative (requires NGC login): nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.1-py3
FROM dustynv/pytorch:2.1-r36.2.0

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    libheif-dev \
    pkg-config \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Virtual environment ────────────────────────────────────────────────────────
# --system-site-packages makes the venv inherit the base image's torch/torchvision
# so we don't need to reinstall them (PyPI wheels target x86, not aarch64/Jetson).
# All subsequent pip/python commands resolve to the venv via PATH.
RUN python3 -m venv /app/venv --system-site-packages
ENV PATH=/app/venv/bin:$PATH

# ── Model downloads ────────────────────────────────────────────────────────────
# Install only what's needed to pull models from HuggingFace, before copying
# requirements.txt — this keeps the large model layers from being invalidated
# by routine requirements changes.
# Pin transformers to the same range as requirements.txt — Florence-2's config
# code accesses forced_bos_token_id before parent __init__ sets it on >= 4.47.
RUN pip install --no-cache-dir "transformers>=4.40,<4.47" huggingface_hub timm einops tqdm

# Models are cached here so the container needs no network access at runtime.
ENV HF_HOME=/app/.hf_cache
# Ensure tqdm progress bars flush immediately to Docker build output.
ENV PYTHONUNBUFFERED=1

RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='microsoft/Florence-2-large'); \
print('Florence-2-large downloaded.')"

RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='google/siglip-so400m-patch14-384'); \
print('SigLIP downloaded.')"

RUN python3 -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='xinyu1205/recognize-anything-plus-model', filename='ram_plus_swin_large_14m.pth', local_dir='.'); \
print('RAM++ checkpoint downloaded.')"

# ── Python dependencies ────────────────────────────────────────────────────────
# Copied after model downloads so changes here don't bust the model cache.

COPY src/requirements.txt .
# Use the pre-built pillow-heif wheel. The manylinux aarch64 wheel bundles its
# own libheif (with AV1/AVIF support), which is newer than the 1.12.0 version
# shipped in Ubuntu 22.04 / L4T R36 — building from source against the system
# libheif fails because pillow-heif>=0.21 requires libheif>=1.16.
RUN pip install --no-cache-dir -r requirements.txt

# Install recognize-anything (RAM++) from GitHub — not on PyPI
RUN pip install --no-cache-dir git+https://github.com/xinyu1205/recognize-anything.git

# ── Application ────────────────────────────────────────────────────────────────
COPY src/server.py .

ENV PORT=9100

EXPOSE 9100

# Single worker to avoid duplicating the large model in memory.
# Raise -w if you have enough RAM (each worker loads its own copy).
CMD ["sh", "-c", "exec gunicorn -w 1 -b 0.0.0.0:${PORT} --timeout 300 server:app"]