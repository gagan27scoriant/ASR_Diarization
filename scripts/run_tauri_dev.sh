#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/tauri_app"

if [[ -f "$HOME/.cargo/env" ]]; then
  # Ensure cargo is on PATH for tauri dev runs.
  . "$HOME/.cargo/env"
fi

if [[ ! -d node_modules ]]; then
  echo "Installing Tauri CLI..."
  npm install
fi

npm run tauri:dev
