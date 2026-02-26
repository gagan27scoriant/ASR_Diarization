# ASR Studio

ASR Studio is a Flask-based speech platform for:
- audio/video transcription
- speaker diarization and speaker mapping
- live meeting chunk transcription
- meeting summary generation
- multilingual translation with local/offline NLLB
- history management and export

## Features

- Upload audio or video and get timestamped transcript with speakers.
- Run live meeting transcription with start/pause/stop controls.
- Extract text from documents (`.pdf`, `.docx`, `.txt`) and summarize it.
- Auto-translate transcript, summary, and document text from the UI language selector.
- Persist sessions in `outputs/*.json` with history list, rename, and delete.
- Export transcript and summary as `.doc` files from the UI.

## Tech Stack

- Backend: Flask
- ASR: `faster-whisper`
- Diarization: `pyannote.audio`
- Translation: NLLB (`transformers`, `sentencepiece`)
- Summarization: Ollama (default) or local BART
- Media/document tools: `ffmpeg-python`, `moviepy`, `pypdf`, `python-docx`

## Requirements

- Python 3.10+ (project is currently used with Conda envs)
- FFmpeg installed on system path
- Optional NVIDIA GPU for faster ASR/translation/diarization
- Hugging Face token for diarization model download (`HUGGINGFACE_TOKEN`)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Open:

```text
http://127.0.0.1:5000
```

## Translation Model (NLLB)

The app prefers local/offline model folders first:
1. `NLLB_MODEL_PATH` (if set)
2. `./nllb_model` (if present)
3. `./models/nllb-200-distilled-600M`
4. Hugging Face cache snapshot (offline)

Optional downloader script:

```bash
python scripts/download_nllb_model.py --output-dir models/nllb-200-distilled-600M --write-env-file .env.nllb
```

Then:

```bash
set -a; source .env.nllb; set +a
```

## Main API Endpoints

- `POST /process` - Process uploaded or path-based media file.
- `POST /transcribe_chunk` - Live chunk transcription.
- `POST /summarize_text` - Summarize transcript text.
- `POST /process_document` - Upload and summarize document.
- `POST /translate` - Translate `text` or `texts`.
- `GET /history` - List session history.
- `GET /history/<session_id>` - Fetch one session.
- `POST /history/<session_id>/transcript` - Save transcript/summary.
- `PATCH /history/<session_id>` - Rename history entry.
- `DELETE /history/<session_id>` - Delete history entry.

## Important Folders

- `audio/` - uploaded and processed audio files
- `videos/` - uploaded videos
- `documents/` - uploaded docs
- `outputs/` - session JSON history files
- `recordings/` - temporary live chunk files
- `demucs_outputs/` - Demucs separation output
- `models/` / `nllb_model/` - local NLLB model folders

## Configuration

See `COMPLETE_DOCUMENTATION.txt` for the full environment variable reference and endpoint payload schemas.

## Troubleshooting

- `NLLB model not found offline`: set `NLLB_MODEL_PATH` to a complete local folder containing model weights.
- `Cannot copy out of meta tensor`: ensure model directory is complete; current loader already handles this and reports incomplete model directories.
- Diarization load failures: set valid `HUGGINGFACE_TOKEN`.
- No summary from Ollama: verify Ollama service and `OLLAMA_URL`.

