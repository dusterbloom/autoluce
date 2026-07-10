#!/usr/bin/env bash
# Convenience wrapper to run autoluce inside the deterministic container.
# Mounts the local work/ directory so downloaded models and build artifacts persist.
set -euo pipefail

IMAGE_TAG="${AUTOLUCE_IMAGE:-autoluce}"

if ! docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
    echo "Building image $IMAGE_TAG..."
    docker build -t "$IMAGE_TAG" "$(dirname "$0")/.."
fi

exec docker run --rm -it \
    -v "$(pwd)/work:/app/work" \
    -e AUTOLUCE_BENCHMARKS="${AUTOLUCE_BENCHMARKS:-}" \
    "$IMAGE_TAG" "$@"
