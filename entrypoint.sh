#!/bin/bash
# Downloads models into the persistent volumes on first run.
# On subsequent starts the presence checks short-circuit immediately.
set -e

HF_HUB=/app/.hf_cache/hub

download_hf_model() {
    local dir="$HF_HUB/$1"
    local repo="$2"
    if [ -d "$dir" ]; then
        echo "[model-init] $repo already present, skipping."
    else
        echo "[model-init] Downloading $repo ..."
        python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$repo')"
    fi
}

download_hf_model "models--microsoft--Florence-2-large"          "microsoft/Florence-2-large"
download_hf_model "models--google--siglip-so400m-patch14-384"    "google/siglip-so400m-patch14-384"
download_hf_model "models--ml6team--keyphrase-extraction-kbir-openkp" "ml6team/keyphrase-extraction-kbir-openkp"

RAM_CKPT=/app/checkpoints/ram_plus_swin_large_14m.pth
if [ -f "$RAM_CKPT" ]; then
    echo "[model-init] RAM++ checkpoint already present, skipping."
else
    echo "[model-init] Downloading RAM++ checkpoint ..."
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='xinyu1205/recognize-anything-plus-model',
                filename='ram_plus_swin_large_14m.pth',
                local_dir='/app/checkpoints')
"
fi

exec "$@"
