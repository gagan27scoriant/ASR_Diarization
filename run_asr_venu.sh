#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

APP_NAME="ASR Venu"

LOG_DIR=""
for candidate in \
  "${XDG_STATE_HOME:-}" \
  "${HOME}/.local/state" \
  "${XDG_CACHE_HOME:-}" \
  "${HOME}/.cache" \
  "$(pwd)/logs"
do
  [[ -z "$candidate" ]] && continue
  if mkdir -p "${candidate}/asr_venu" >/dev/null 2>&1; then
    LOG_DIR="${candidate}/asr_venu"
    break
  fi
done
if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="$(pwd)/logs"
  mkdir -p "$LOG_DIR" >/dev/null 2>&1 || true
fi

LOG_FILE="${LOG_DIR}/launcher-$(date +%Y%m%d-%H%M%S).log"

show_error() {
  local text="$1"
  if command -v zenity >/dev/null 2>&1 && [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
    zenity --error --title="$APP_NAME" --text="$text" >/dev/null 2>&1 || true
    return 0
  fi
  if command -v notify-send >/dev/null 2>&1; then
    notify-send --urgency=critical "$APP_NAME" "$text" >/dev/null 2>&1 || true
    return 0
  fi
  if command -v xmessage >/dev/null 2>&1 && [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
    xmessage -center "$text" >/dev/null 2>&1 || true
    return 0
  fi
  printf '%s\n' "$text" >&2
}

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" && -x "./.venv/bin/python" ]]; then
  PYTHON_BIN="./.venv/bin/python"
fi
if [[ -z "$PYTHON_BIN" && -x "/home/scoriant/miniconda3/bin/python" ]]; then
  PYTHON_BIN="/home/scoriant/miniconda3/bin/python"
fi
if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

export PYWEBVIEW_GUI="${PYWEBVIEW_GUI:-qt}"
export QT_API="${QT_API:-pyqt5}"

{
  echo "=== ${APP_NAME} launcher ==="
  echo "Time: $(date -Is)"
  echo "CWD: $(pwd)"
  echo "Python: $PYTHON_BIN"
  echo "PYWEBVIEW_GUI: ${PYWEBVIEW_GUI}"
  echo "QT_API: ${QT_API}"
  echo
} >>"$LOG_FILE" 2>&1

if ! "$PYTHON_BIN" -c 'import torch, torchaudio; print("torch", torch.__version__); print("torchaudio", torchaudio.__version__)' >>"$LOG_FILE" 2>&1; then
  show_error $'ASR Venu failed to start.\n\nReason: Python could not import torch/torchaudio.\n\nOpen the log for details:\n'"$LOG_FILE"$'\n\nFix: reinstall matching torch + torchaudio versions (same 2.x release) in the Python environment used by this launcher.'
  exit 1
fi

if ! "$PYTHON_BIN" asr_venu.py >>"$LOG_FILE" 2>&1; then
  show_error $'ASR Venu exited with an error.\n\nOpen the log for details:\n'"$LOG_FILE"
  exit 1
fi
