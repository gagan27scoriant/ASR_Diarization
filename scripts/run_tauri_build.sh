#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/tauri_app"

if [[ -f "$HOME/.cargo/env" ]]; then
  # Ensure cargo is on PATH for tauri builds.
  . "$HOME/.cargo/env"
fi

if [[ ! -d node_modules ]]; then
  echo "Installing Tauri CLI..."
  npm install
fi

# AppImage-based tools (linuxdeploy) need FUSE; extract-and-run avoids that.
export APPIMAGE_EXTRACT_AND_RUN=1
# If a runtime file is provided locally, use it to avoid network fetches.
RUNTIME_FILE="$ROOT_DIR/tauri_app/src-tauri/runtime-x86_64"
if [[ -f "$RUNTIME_FILE" ]]; then
  export LDAI_RUNTIME_FILE="$RUNTIME_FILE"
  export APPIMAGE_RUNTIME_FILE="$RUNTIME_FILE"
fi

npm run tauri:build
