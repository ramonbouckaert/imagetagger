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

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── CUDA runtime libraries + cuDNN 9 ──────────────────────────────────────────
# cuda-libraries-12-6 is a meta-package that pulls in the full set of CUDA 12.6
# runtime libs (cublas, cufft, curand, cusolver, cusparse, cupti, etc.) that
# PyTorch links against. cuDNN is installed from the same repo.
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        cuda-libraries-12-6 \
        libcudnn9-cuda-12 \
        libcudnn9-dev-cuda-12 \
    && rm -rf /var/lib/apt/lists/* cuda-keyring_1.1-1_all.deb

# ── cuSPARSELt ────────────────────────────────────────────────────────────────
# Must be installed via .deb before torch, as the wheel depends on the shared libs.
RUN wget -q https://developer.download.nvidia.com/compute/cusparselt/0.7.0/local_installers/cusparselt-local-tegra-repo-ubuntu2204-0.7.0_1.0-1_arm64.deb \
    && dpkg -i cusparselt-local-tegra-repo-ubuntu2204-0.7.0_1.0-1_arm64.deb \
    && cp /var/cusparselt-local-tegra-repo-ubuntu2204-0.7.0/cusparselt-*-keyring.gpg /usr/share/keyrings/ \
    && apt-get update \
    && apt-get install -y --no-install-recommends libcusparselt0 libcusparselt-dev \
    && rm -rf /var/lib/apt/lists/* cusparselt-local-tegra-repo-ubuntu2204-0.7.0_1.0-1_arm64.deb

# ── Virtual environment ────────────────────────────────────────────────────────
RUN python3 -m venv /app/venv
ENV PATH=/app/venv/bin:$PATH

# ── PyTorch from Jetson AI Lab wheels ─────────────────────────────────────────
# Same wheels as bare-metal install; built for JetPack 6 / CUDA 12.6 / Python 3.10.
RUN pip install --no-cache-dir \
    --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
    torch==2.10.0 \
    torchvision==0.25.0

# ── Python dependencies ────────────────────────────────────────────────────────
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install recognize-anything (RAM++) from GitHub — not on PyPI
RUN pip install --no-cache-dir \
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