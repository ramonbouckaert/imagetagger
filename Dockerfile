# Target: NVIDIA Jetson Orin (JetPack 6.2 / L4T R36.4 / CUDA 12.6).
# PyTorch and torchvision are installed from the Jetson AI Lab PyPI index,
# which provides aarch64 wheels built for JetPack 6 — the same wheels that
# work on bare metal.
FROM nvcr.io/nvidia/l4t-cuda:12.6.11-runtime

# ── APT mirror ────────────────────────────────────────────────────────────────
# Handles both Ubuntu 20.04 (sources.list) and 24.04 (DEB822 .sources format).
RUN sed -i 's|http://ports.ubuntu.com/ubuntu-ports|http://mirror.aarnet.edu.au/ubuntu-ports|g' /etc/apt/sources.list /etc/apt/sources.list.d/*.list 2>/dev/null; \
    sed -i 's|http://ports.ubuntu.com/ubuntu-ports|http://mirror.aarnet.edu.au/ubuntu-ports|g' /etc/apt/sources.list.d/*.sources 2>/dev/null; \
    true

# ── JetPack apt repo (L4T r36.4 / JetPack 6.2) ───────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg \
    && curl -fsSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc \
       | gpg --dearmor -o /usr/share/keyrings/nvidia-jetson.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/nvidia-jetson.gpg] \
       https://repo.download.nvidia.com/jetson/common r36.4 main" \
       > /etc/apt/sources.list.d/nvidia-jetson.list \
    && echo "deb [signed-by=/usr/share/keyrings/nvidia-jetson.gpg] \
       https://repo.download.nvidia.com/jetson/t234 r36.4 main" \
       > /etc/apt/sources.list.d/nvidia-jetson-t234.list \
    && apt-get update \
    && rm -rf /var/lib/apt/lists/*

# ── NVIDIA CUDA apt repo keyring ──────────────────────────────────────────────
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb -o cuda-keyring.deb \
    && dpkg -i cuda-keyring.deb \
    && rm cuda-keyring.deb \
    && apt-get update \
    && rm -rf /var/lib/apt/lists/*

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends --allow-change-held-packages \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    pkg-config \
    gcc \
    g++ \
    python3-pip \
    python3-venv \
    libopenblas0 \
    libnuma1 \
    libjpeg-turbo8 \
    libpng16-16 \
    libavif-dev \
    python3-dev \
    libvips-dev \
    cuda-libraries-12-6 \
    cuda-cupti-12-6 \
    libnvjitlink-12-6 \
    cuda-nvtx-12-6 \
    libcublas-12-6 \
    libcufft-12-6 \
    libcurand-12-6 \
    libcusolver-12-6 \
    libcusparse-12-6 \
    cuda-nvrtc-12-6 \
    cudss \
    libcudnn9-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── cuSPARSELt ────────────────────────────────────────────────────────────────
# Must be installed via .deb before torch, as the wheel depends on the shared libs.
RUN wget -q https://developer.download.nvidia.com/compute/cusparselt/0.9.0/local_installers/cusparselt-local-repo-ubuntu2204-0.9.0_0.9.0-1_arm64.deb \
    && dpkg -i cusparselt-local-repo-ubuntu2204-0.9.0_0.9.0-1_arm64.deb \
    && cp /var/cusparselt-local-repo-ubuntu2204-0.9.0/cusparselt-*-keyring.gpg /usr/share/keyrings/ \
    && apt-get update \
    && apt-get install -y --no-install-recommends libcusparselt0 libcusparselt-dev \
    && rm -rf /var/lib/apt/lists/* cusparselt-local-repo-ubuntu2204-0.9.0_0.9.0-1_arm64.deb

# ── Virtual environment ────────────────────────────────────────────────────────
RUN python3 -m venv /app/venv
ENV PATH=/app/venv/bin:$PATH

# ── PyTorch from Jetson AI Lab wheels ─────────────────────────────────────────
# Same wheels as bare-metal install; built for JetPack 6 / CUDA 12.6 / Python 3.10.
RUN pip install --no-cache-dir \
    --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
    torch==2.11.0 \
    torchvision==0.26.0

# ── Python dependencies ────────────────────────────────────────────────────────
COPY src/requirements.txt .
RUN pip install --no-cache-dir --no-binary pillow -r requirements.txt

# Install recognize-anything (RAM++) from GitHub — not on PyPI
RUN pip install --no-cache-dir \
    git+https://github.com/xinyu1205/recognize-anything.git

# Patch RAM++
COPY patches/ram_bert.py /app/venv/lib/python3.10/site-packages/ram/models/bert.py
COPY patches/ram_utils.py /app/venv/lib/python3.10/site-packages/ram/models/utils.py

# ── Application ────────────────────────────────────────────────────────────────
COPY src/config.py src/models.py src/controller.py src/server.py .
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
# gthread worker lets multiple requests queue concurrently within the one process;
# inference serialisation is handled by each model's own executor.
CMD ["sh", "-c", "exec gunicorn -w 1 --worker-class gthread --threads 16 -b 0.0.0.0:${PORT} --timeout 300 server:app"]
