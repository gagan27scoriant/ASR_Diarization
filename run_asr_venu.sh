#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" && -x "/home/scoriant/miniconda3/bin/python" ]]; then
  PYTHON_BIN="/home/scoriant/miniconda3/bin/python"
fi
if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

export PYWEBVIEW_GUI="${PYWEBVIEW_GUI:-qt}"
export QT_API="${QT_API:-pyqt5}"

"$PYTHON_BIN" asr_venu.py
