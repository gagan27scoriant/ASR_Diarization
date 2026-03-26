# AI Knowledge Studio 🧠✨

AI Knowledge Studio 🧠✨ is a Flask-based speech + document intelligence platform for:
- audio/video transcription with diarization
- live meeting chunk transcription
- meeting summary generation
- document ingestion + RAG Q&A
- multilingual translation with local/offline NLLB
- session history and document history

## Features

- Upload audio or video and get timestamped transcript with speakers.
- Run live meeting transcription with start/pause/stop controls.
- Upload documents (`.pdf`, `.docx`, `.txt`) to extract text, summarize, and ask questions.
- RAG Q&A over documents with history-aware responses.
- Auto-translate transcript, summary, and document text from the UI language selector.
- Persist history in MongoDB (sessions + documents) with list, open, rename, delete.
- Export transcript and summary as `.doc` files from the UI.

## Tech Stack

- Backend: Flask
- Database: MongoDB (history + document store)
- ASR: `faster-whisper`
- Diarization: `pyannote.audio`
- Translation: NLLB (`transformers`, `sentencepiece`)
- Text-to-speech: gTTS
- Summarization: Ollama (default) or local BART
- Document tools: `pypdf`, `python-docx`, optional OCR (`pdf2image`, `pytesseract`)
- RAG embeddings: `sentence-transformers`

## Requirements

- Python 3.10+ (project is currently used with Conda envs)
- MongoDB running locally (or set `MONGODB_URI`)
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

## Document Q&A (RAG)

- Upload a PDF/DOCX/TXT and the app will:
  - extract text
  - split into chunks (recursive splitter)
  - embed and store chunks in MongoDB
  - enable document Q&A in the UI

Document history appears in the sidebar with a mini PDF preview. Use View/Zoom in the sidebar to open the PDF.

### RAG Environment Variables

- `RAG_MODEL` (default: `mistral`) – model used by Ollama for Q&A
- `DOC_CHUNK_SIZE` (default: `2000`)
- `DOC_CHUNK_OVERLAP` (default: `80`)
- `DOC_RETRIEVE_TOP_K` (default: `5`)

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
- `POST /process_document` - Upload and summarize document + RAG ingestion.
- `POST /api/document/ask` - Ask a question about a document.
- `GET /api/documents` - List document history.
- `GET /api/documents/<doc_id>` - Open a document history item.
- `POST /translate` - Translate `text` or `texts`.
- `GET /api/agent/tools` - List the tools exposed to the agent router.
- `POST /api/agent/query` - Query-first endpoint that selects a tool based on intent and context.
- `POST /api/agent/chat` - Unified agent endpoint for uploaded files plus query-driven execution.
- `GET /history` - List session history.
- `GET /history/<session_id>` - Fetch one session.
- `POST /history/<session_id>/transcript` - Save transcript/summary.
- `PATCH /history/<session_id>` - Rename history entry.
- `DELETE /history/<session_id>` - Delete history entry.

## Important Folders

- `audio/` - uploaded and processed audio files
- `videos/` - uploaded videos
- `documents/` - uploaded docs
- `recordings/` - temporary live chunk files
- `demucs_outputs/` - Demucs separation output
- `models/` / `nllb_model/` - local NLLB model folders

## Agent Query Mode

You can now call the app in a query-first way instead of choosing fixed endpoints yourself. The new agent layer maps your query onto existing tools such as:

- media processing
- transcript summarization
- translation
- transcript Q&A
- document Q&A
- transcript semantic search
- text-to-speech

Example:

```bash
curl -X POST http://127.0.0.1:5000/api/agent/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "query": "summarize this meeting and translate it to Hindi",
    "content": "Speaker 1: ...",
    "meeting_title": "Weekly Review",
    "meeting_date": "2026-03-26",
    "meeting_place": "Conference Room",
    "target_lang": "hin_Deva"
  }'
```

The route returns:

- the selected tool
- the execution plan
- the tool result

## gTTS Note

`gTTS` is now available through the agent as a text-to-speech tool. Generated speech files are saved into `audio/` and can be played back from the UI.

At runtime, `gTTS` typically requires internet access to synthesize speech.

## Configuration

See `COMPLETE_DOCUMENTATION.txt` for the full environment variable reference and endpoint payload schemas.

## Troubleshooting

- Mongo error `document too large`: resolved by chunk storage in `document_chunks` collection. Ensure Mongo is running.
- `NLLB model not found offline`: set `NLLB_MODEL_PATH` to a complete local folder containing model weights.
- Diarization load failures: set valid `HUGGINGFACE_TOKEN`.
- No summary from Ollama: verify Ollama service and `OLLAMA_URL`.
