FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────────
WORKDIR /app

COPY src/requirements.txt .

# Install torch CPU-only first (smaller image — swap the index URL for GPU builds)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining deps
RUN pip install --no-cache-dir -r requirements.txt

# ── Download models at build time ─────────────────────────────────────────────
# Models are cached here so the container needs no network access at runtime.
ENV HF_HOME=/app/.hf_cache

RUN python3 -c "\
from transformers import AutoProcessor, AutoModelForCausalLM; \
AutoProcessor.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True); \
AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True); \
print('Florence-2-large downloaded.')"

ARG SIGLIP_CACHE_BUST=1
RUN python3 -c "\
from transformers import SiglipModel, SiglipProcessor; \
SiglipModel.from_pretrained('google/siglip-so400m-patch14-384'); \
SiglipProcessor.from_pretrained('google/siglip-so400m-patch14-384'); \
print('SigLIP downloaded.')"

# ── Application ────────────────────────────────────────────────────────────────
COPY src/server.py .

ENV PORT=9100

EXPOSE 9100

# Single worker to avoid duplicating the large model in memory.
# Raise -w if you have enough RAM (each worker loads its own copy).
CMD ["sh", "-c", "exec gunicorn -w 1 -b 0.0.0.0:${PORT} --timeout 120 server:app"]