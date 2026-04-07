#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PID_FILE="/tmp/asr_venu_flask.pid"
LOG_FILE="/tmp/asr_venu_flask.log"
APP_PORT="${APP_PORT:-1627}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" && -x "/home/scoriant/miniconda3/bin/python" ]]; then
  PYTHON_BIN="/home/scoriant/miniconda3/bin/python"
fi
if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

# Keep backend selection explicit for Tauri runs unless user overrides it.
export DIARIZATION_BACKEND="${DIARIZATION_BACKEND:-auto}"
export APP_PORT
export DOCUMENT_MAX_PDF_PAGES="${DOCUMENT_MAX_PDF_PAGES:-80}"
export DOCUMENT_MAX_CHARS="${DOCUMENT_MAX_CHARS:-180000}"
export DOC_MAX_CHUNKS="${DOC_MAX_CHUNKS:-90}"
export DOC_TRANSLATE_MAX_SEGMENTS="${DOC_TRANSLATE_MAX_SEGMENTS:-90}"
export DOCUMENT_SUMMARY_MAX_CHARS="${DOCUMENT_SUMMARY_MAX_CHARS:-20000}"

if [[ -f "$PID_FILE" ]]; then
  EXISTING_PID="$(cat "$PID_FILE" || true)"
  if [[ -n "$EXISTING_PID" ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
    echo "Flask already running (pid=$EXISTING_PID)."
  else
    rm -f "$PID_FILE"
  fi
fi

if [[ ! -f "$PID_FILE" ]]; then
  echo "Using Python: $PYTHON_BIN" > "$LOG_FILE"
  echo "APP_PORT=$APP_PORT" >> "$LOG_FILE"
  echo "DIARIZATION_BACKEND=$DIARIZATION_BACKEND" >> "$LOG_FILE"
  echo "DOCUMENT_MAX_PDF_PAGES=$DOCUMENT_MAX_PDF_PAGES" >> "$LOG_FILE"
  echo "DOC_MAX_CHUNKS=$DOC_MAX_CHUNKS" >> "$LOG_FILE"
  echo "DOCUMENT_SUMMARY_MAX_CHARS=$DOCUMENT_SUMMARY_MAX_CHARS" >> "$LOG_FILE"
  nohup "$PYTHON_BIN" main.py >> "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  echo "Started Flask (pid=$(cat "$PID_FILE"))."
fi

python - <<'PY'
import os
import time
import urllib.request
import urllib.error

port = int((os.getenv("APP_PORT") or "1627").strip())
url = f"http://127.0.0.1:{port}/"
ready_codes = {200, 401, 403}

for _ in range(240):
    try:
        with urllib.request.urlopen(url, timeout=1) as resp:
            if resp.status in ready_codes:
                break
    except urllib.error.HTTPError as err:
        if err.code in ready_codes:
            break
    except Exception:
        pass
    time.sleep(0.5)
else:
    raise SystemExit("Flask server did not become ready in time.")
PY
