#!/bin/bash
set -e
REPO="chathedev/kbgpu"
TAG="latest"
IMAGE="ghcr.io/$REPO:$TAG"
echo "Building $IMAGE..."
docker buildx build --platform linux/amd64 -t $IMAGE . --push
echo "Done: $IMAGE"
