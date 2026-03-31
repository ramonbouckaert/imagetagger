# Target: NVIDIA Jetson Orin (JetPack 6 / L4T R36).
# PyTorch 2.7 with CUDA is pre-installed in this base image — do NOT reinstall
# torch/torchvision from PyPI; the standard wheels target x86 and will not work
# on aarch64/Jetson.
#
# Using dusty-nv's image (Docker Hub, no auth required).
# Official NVIDIA alternative (requires NGC login): nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.1-py3
FROM dustynv/pytorch:2.7-r36.4.0

# ── APT mirror ────────────────────────────────────────────────────────────────
RUN sed -i 's|http://ports.ubuntu.com/ubuntu-ports|http://mirror.aarnet.edu.au/ubuntu-ports|g' /etc/apt/sources.list

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    pkg-config \
    gcc \
    g++ \
    python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Virtual environment ────────────────────────────────────────────────────────
# --system-site-packages makes the venv inherit the base image's torch/torchvision
# so we don't need to reinstall them (PyPI wheels target x86, not aarch64/Jetson).
# All subsequent pip/python commands resolve to the venv via PATH.
RUN python3 -m venv /app/venv --system-site-packages
ENV PATH=/app/venv/bin:$PATH

# Pin torch to the exact version already in the base image so that no subsequent
# 'pip install' can upgrade it to an incompatible PyPI wheel.
RUN python3 -c "\
import torch; \
open('/constraints.txt','w').write(f'torch=={torch.__version__}\n'); \
print('Pinned:', open('/constraints.txt').read().strip())" \
    || (echo "torch not importable in venv — skipping pin" && touch /constraints.txt)

# ── Python dependencies ────────────────────────────────────────────────────────
COPY src/requirements.txt .
RUN pip install --no-cache-dir --constraint /constraints.txt -r requirements.txt

# Install recognize-anything (RAM++) from GitHub — not on PyPI
RUN pip install --no-cache-dir --constraint /constraints.txt \
    git+https://github.com/xinyu1205/recognize-anything.git

# ── Application ────────────────────────────────────────────────────────────────
COPY src/server.py .
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Models are downloaded by the entrypoint into the persistent volumes on first
# run and skipped on all subsequent starts (including after image rebuilds).
ENV HF_HOME=/app/.hf_cache
ENV PYTHONUNBUFFERED=1
ENV PORT=9100

EXPOSE 9100

ENTRYPOINT ["/entrypoint.sh"]
# Single worker to avoid duplicating the large model in memory.
CMD ["sh", "-c", "exec gunicorn -w 1 -b 0.0.0.0:${PORT} --timeout 300 server:app"]
