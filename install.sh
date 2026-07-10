#!/usr/bin/env bash
# AutoLuce one-liner installer.
#
#   curl -fsSL https://raw.githubusercontent.com/dusterbloom/autoluce/main/install.sh | bash
#
# Clones the repo, ensures `uv` is installed, syncs dependencies, and prints next steps.
# Override the source with AUTOLUCE_REPO_URL / AUTOLUCE_DEST. Lucebox product builds
# currently require a CUDA or HIP toolkit.
set -euo pipefail

REPO_URL="${AUTOLUCE_REPO_URL:-https://github.com/dusterbloom/autoluce.git}"
DEST="${AUTOLUCE_DEST:-autoluce}"

echo "==> AutoLuce installer"
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
    uv run autoluce source status            # inspect the pinned Lucebox contract
    uv run autoluce source check --remote    # verify the Hub pin is current
    uv run autoluce reproduce --simulate     # test the control plane without a GPU
    uv run autoluce setup                    # clone/build product (CUDA or HIP)
    uv run autoluce help                     # all commands
NEXT
