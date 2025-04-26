#!/bin/bash

# Set image name
IMAGE="nvcr.io/qhrjhjrvlsbu/aerial-cuda-accelerated-ran:24-3-cubb"
SRC_DIR="/opt/nvidia/cuBB"

# Set destination directory (default: current directory)
DEST_DIR="${1:-./cuBB}"

mkdir -p $DEST_DIR

# Create a temporary container
CONTAINER_ID=$(docker create "$IMAGE")

# Copy directory from container to host
docker cp "$CONTAINER_ID:$SRC_DIR" "$DEST_DIR"

# Remove temporary container
docker rm "$CONTAINER_ID"

echo "Directory copied to $DEST_DIR"


# cd $cuBB_SDK/pyaerial/notebooks
# jupyter lab --ip=0.0.0.0