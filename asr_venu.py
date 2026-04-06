#!/usr/bin/env python3
"""Launch ASR Venu in a desktop window using pywebview."""
import os
import threading
import time
import urllib.request
import urllib.error

import webview

from main import app

APP_PORT = int((os.getenv("APP_PORT") or "1627").strip())
APP_URL = f"http://127.0.0.1:{APP_PORT}"

def _run_server():
    # Avoid reloader in embedded mode
    app.run(host="127.0.0.1", port=APP_PORT, debug=False, use_reloader=False)


def _wait_for_server(url: str, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as _:
                return True
        except urllib.error.HTTPError as e:
            # Auth-required is still a valid server response.
            if e.code in {401, 403}:
                return True
            time.sleep(0.3)
            continue
        except Exception:
            time.sleep(0.3)
    return False


def main():
    # Run Flask in background
    thread = threading.Thread(target=_run_server, daemon=True)
    thread.start()

    url = APP_URL
    _wait_for_server(url)

    webview.create_window("ASR Venu", url, width=1280, height=800)
    gui = os.getenv("PYWEBVIEW_GUI", "gtk")
    try:
        webview.start(gui=gui, debug=False)
    except Exception as e:
        print(f"Failed to start native window with GUI '{gui}': {e}")
        print("Install a supported GUI backend (gtk or qt) and try again.")


if __name__ == "__main__":
    main()
