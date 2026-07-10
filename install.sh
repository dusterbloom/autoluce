#!/usr/bin/env bash
# autoggml one-liner installer.
#
#   curl -fsSL https://raw.githubusercontent.com/dusterbloom/autoggml/main/install.sh | bash
#
# Clones the repo, ensures `uv` is installed, syncs dependencies, and prints next steps.
# Override the source with AUTOGGML_REPO_URL / AUTOGGML_DEST. Lucebox product builds
# currently require a CUDA or HIP toolkit.
set -euo pipefail

REPO_URL="${AUTOGGML_REPO_URL:-https://github.com/dusterbloom/autoggml.git}"
DEST="${AUTOGGML_DEST:-autoggml}"

echo "==> autoggml installer"
echo "    repo: $REPO_URL"
echo "    dest: $DEST"

if ! command -v git >/dev/null 2>&1; then
  echo "error: git is required (install git and re-run)" >&2
  exit 1
fi

if [ -e "$DEST" ]; then
  echo "==> '$DEST' exists; pulling latest"
  git -C "$DEST" pull --ff-only
else
  git clone --depth 1 "$REPO_URL" "$DEST"
fi

cd "$DEST"

# Ensure uv (the project manager). uv installs to ~/.local/bin; put it on PATH if missing.
if ! command -v uv >/dev/null 2>&1; then
  echo "==> installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> uv sync (resolving Python + dependencies)"
uv sync

cat <<'NEXT'

==> installed. Next:
    uv run autoggml source status            # inspect the pinned Lucebox contract
    uv run autoggml source check --remote    # verify the Hub pin is current
    uv run autoggml reproduce --simulate     # test the control plane without a GPU
    uv run autoggml setup                    # clone/build product (CUDA or HIP)
    uv run autoggml help                     # all commands
NEXT
