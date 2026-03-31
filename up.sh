#!/bin/bash
# Starts imagetagger, selecting the best base image for the current JetPack version.
# Requires jetson-containers to be installed: https://github.com/dusty-nv/jetson-containers
set -e

if command -v autotag &>/dev/null; then
    export BASE_IMAGE=$(autotag l4t-pytorch)
    echo "Using base image: $BASE_IMAGE"
else
    echo "Warning: autotag not found — using default base image from Dockerfile."
fi

exec docker compose up --build "$@"