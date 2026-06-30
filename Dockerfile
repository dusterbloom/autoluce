# Deterministic environment for autoggml v2 experiments.
#
# Build:
#   docker build -t autoggml .
#
# Run baseline harness in container:
#   docker run --rm -it -v $(pwd)/work:/app/work autoggml uv run harness.py --baseline
#
# Run full reproduction suite:
#   docker run --rm -it -v $(pwd)/work:/app/work autoggml uv run reproduce.py

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Build dependencies for lucebox-ggml / llama.cpp.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    ccache \
    git \
    curl \
    ca-certificates \
    python3-dev \
    time \
    && rm -rf /var/lib/apt/lists/*

# Install uv at a pinned version for reproducibility.
ARG UV_VERSION=0.5.5
ADD https://astral.sh/uv/${UV_VERSION}/install.sh /tmp/uv-install.sh
RUN sh /tmp/uv-install.sh && rm /tmp/uv-install.sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Copy dependency manifest and lockfile first to leverage Docker layer caching.
COPY pyproject.toml .python-version uv.lock README.md ./
RUN uv sync --frozen

# Copy source code.
COPY . .

# Default: run lint and smoke tests.
CMD ["uv", "run", "reproduce.py"]
