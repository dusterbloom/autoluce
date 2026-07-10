# Deterministic environment for autoggml v2 experiments.
#
# Build:
#   docker build -t autoggml .
#
# Run baseline harness in container:
#   docker run --rm -it -v $(pwd)/work:/app/work autoggml uv run autoggml baseline
#
# Run full reproduction suite:
#   docker run --rm -it -v $(pwd)/work:/app/work autoggml uv run autoggml reproduce

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Build dependencies for Lucebox Hub and its vendored GGML.
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

# Copy dependency manifest and lockfile first to leverage Docker layer caching:
# install only third-party deps here (the autoggml package itself doesn't exist yet).
COPY pyproject.toml .python-version uv.lock README.md ./
RUN uv sync --frozen --no-install-project

# Copy source code, then install the project into the environment.
COPY . .
RUN uv sync --frozen

# Default: run lint and smoke tests.
CMD ["uv", "run", "autoggml", "reproduce"]
