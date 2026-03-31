#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PID_FILE="/tmp/asr_venu_flask.pid"
LOG_FILE="/tmp/asr_venu_flask.log"

if [[ -f "$PID_FILE" ]]; then
  EXISTING_PID="$(cat "$PID_FILE" || true)"
  if [[ -n "$EXISTING_PID" ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
    echo "Flask already running (pid=$EXISTING_PID)."
  else
    rm -f "$PID_FILE"
  fi
fi

if [[ ! -f "$PID_FILE" ]]; then
  nohup python main.py > "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  echo "Started Flask (pid=$(cat "$PID_FILE"))."
fi

python - <<'PY'
import time
import urllib.request
import urllib.error

url = "http://127.0.0.1:5000/"
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
