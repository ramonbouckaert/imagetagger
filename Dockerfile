# Target: NVIDIA Jetson Orin (JetPack 6.2 / L4T R36.4 / CUDA 12.6).
# PyTorch and torchvision are installed from the Jetson AI Lab PyPI index,
# which provides aarch64 wheels built for JetPack 6 — the same wheels that
# work on bare metal.
FROM dustynv/cuda:12.6-r36.4.0

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    pkg-config \
    gcc \
    g++ \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Virtual environment ────────────────────────────────────────────────────────
RUN python3 -m venv /app/venv
ENV PATH=/app/venv/bin:$PATH

# ── PyTorch from Jetson AI Lab wheels ─────────────────────────────────────────
# Same wheels as bare-metal install; built for JetPack 6 / CUDA 12.6 / Python 3.10.
RUN pip install --no-cache-dir \
    --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
    torch==2.10.0 \
    torchvision==0.25.0 \
    cusparselt

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