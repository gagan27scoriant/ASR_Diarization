#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.translation import DEFAULT_NLLB_LOCAL_DIR, DEFAULT_NLLB_MODEL_NAME, download_nllb_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Download NLLB model locally and print NLLB_MODEL_PATH.")
    parser.add_argument("--model-name", default=DEFAULT_NLLB_MODEL_NAME, help="Hugging Face model id")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_NLLB_LOCAL_DIR,
        help="Local directory where model files will be stored",
    )
    parser.add_argument(
        "--write-env-file",
        default=".env.nllb",
        help="Optional env file to write NLLB_MODEL_PATH export values",
    )
    args = parser.parse_args()

    local_path = download_nllb_model(model_name=args.model_name, target_dir=args.output_dir)
    env_file = (args.write_env_file or "").strip()
    if env_file:
        with open(env_file, "w", encoding="utf-8") as f:
            f.write(f"NLLB_MODEL_PATH={local_path}\n")
            f.write(f"NLLB_MODEL_NAME={args.model_name}\n")
            f.write("NLLB_OFFLINE_ONLY=1\n")
        print(f"Wrote environment file: {os.path.abspath(env_file)}")

    print(f"Use this in your shell before starting the app:\nexport NLLB_MODEL_PATH='{local_path}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
