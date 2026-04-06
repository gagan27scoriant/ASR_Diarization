# AI Knowledge Studio

AI Knowledge Studio is a Flask-based speech, document, and query-first assistant workspace. It supports:

- audio/video transcription with speaker diarization
- live meeting chunk transcription
- transcript and meeting summarization
- document upload, summarization, and RAG Q&A
- translation with local/offline NLLB
- text-to-speech with gTTS (plus browser fallback)
- query-driven agent routing
- sidebar history for transcript sessions and documents
- plain chatbot mode with conversation history
- deterministic utility handling for date/time and math-style queries

## Current Product Behavior

The app has two main interaction styles:

1. Plain chat mode
   - If you type a query without uploading a file or opening a transcript/document context, the assistant behaves like a normal chatbot.
   - Plain chat is history-aware within the current browser session.
   - Simple date/time questions are answered from runtime code instead of model guessing.
   - Many math questions are solved deterministically with `sympy`.

2. Agent / tool mode
   - If your query requires app workflows such as processing media, summarizing, translating, document Q&A, transcript Q&A, search, or text-to-speech, the agent router selects the matching tool.
   - If you attach a file, the app waits for your query first, then processes the file according to your instruction.

## Features

- Upload audio/video and get timestamped transcripts with speakers.
- Start, pause, and stop live meeting capture.
- Upload PDF/DOCX/TXT and ask questions about them.
- Generate structured meeting summaries for transcripts.
- Generate concise summaries for documents.
- Translate transcript text, summaries, and document-derived text.
- Convert supported text content into speech.
- Use transcript semantic search / keyword-like search.
- Reopen, rename, and delete transcript history.
- Reopen and manage document history.
- Export transcript and summary as **PDF** in the minutes format.

## Tech Stack

- Backend: Flask
- Database: MongoDB
- ASR: `faster-whisper`
- Diarization: `pyannote.audio` (default) or NVIDIA NeMo (optional)
- Summarization / chat / RAG generation: Ollama-backed local model flow
- Translation: NLLB via `transformers`
- Text-to-speech: gTTS (server) + browser SpeechSynthesis fallback
- Embeddings / retrieval: `sentence-transformers`
- Deterministic math: `sympy`

## Requirements

- Python 3.10+
- MongoDB running locally, or set `MONGODB_URI`
- FFmpeg installed and available on PATH
- Ollama running locally if you use LLM-backed chat / summary / RAG flows
- Optional NVIDIA GPU for faster ASR / diarization / translation
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
http://127.0.0.1:1627
```

## Main User Flow

### Plain Chat

- Type into the docked agent/chat bar.
- The assistant responds in chat mode.
- Follow-up chat turns reuse recent chat history in the current session.

### Upload + Query

- Attach a file from the chat bar.
- File upload is staged first.
- The app does not process immediately.
- Type your instruction, then send.
- The agent selects the matching workflow based on the query.

Examples:

- `transcribe and diarize this recording`
- `summarize this meeting and highlight action items`
- `ask this document for the conclusion`
- `translate the summary to Hindi`
- `convert this answer to speech`

## Agent Endpoints

- `GET /api/agent/tools`
  - lists the currently exposed tools

- `POST /api/agent/query`
  - query-first endpoint for chat or tool routing
  - accepts JSON

- `POST /api/agent/chat`
  - unified endpoint for uploaded file + query flow
  - accepts multipart form-data when uploading a file
  - falls back to the same query route when no file is included

## Other Main API Endpoints

- `POST /process`
- `POST /transcribe_chunk`
- `POST /summarize_text`
- `POST /process_document`
- `POST /api/document/ask`
- `POST /api/history/ask`
- `GET /api/documents`
- `GET /api/documents/<doc_id>`
- `POST /translate`
- `GET /history`
- `GET /history/<session_id>`
- `POST /history/<session_id>/transcript`
- `PATCH /history/<session_id>`
- `DELETE /history/<session_id>`

## Agent Tools Currently Exposed

The router can currently select from tools such as:

- `chat_response`
- `process_media`
- `summarize_transcript`
- `translate_text`
- `answer_document`
- `answer_history`
- `search_history`
- `text_to_speech`

Important note:
- the backend is tool-routed and query-first
- but it is still mostly single-tool selection plus a few chained cases
- it is not yet a full autonomous multi-step planning agent for every request

## Chatbot Memory and Deterministic Utilities

Plain chatbot mode currently supports:

- short-term conversation memory in the current UI session
- deterministic answers for common date/time questions
- deterministic handling for many math queries

Examples:

- `wt day is today`
- `what is today's date`
- `what time is it`
- `what is 2+2`
- `calculate 25% of 80`
- `solve x^2 - 5*x + 6 = 0`

## Translation Model

The app prefers local/offline NLLB model folders in this order:

1. `NLLB_MODEL_PATH`
2. `./nllb_model`
3. `./models/nllb-200-distilled-600M`
4. Hugging Face cache snapshot if available offline

Optional downloader:

```bash
python scripts/download_nllb_model.py --output-dir models/nllb-200-distilled-600M --write-env-file .env.nllb
```

Then:

```bash
set -a; source .env.nllb; set +a
```

## Text-to-Speech Note

- gTTS is exposed through the agent/tool path
- generated speech files are written into `audio/`
- gTTS typically requires internet access at runtime
- if gTTS fails, the UI falls back to browser SpeechSynthesis

## Diarization Options

- Default: pyannote diarization.
- Optional: NVIDIA NeMo diarization (see `configs/nemo_diarization.yaml`).

## Export Format

- Summary and Transcript exports download as **PDF**.
- The PDF is structured as **Minutes of a Meeting** with headings and bullets.

## Important Folders

- `audio/`
- `videos/`
- `documents/`
- `recordings/`
- `demucs_outputs/`
- `models/`
- `nllb_model/`

## Documentation Files

- `README.md`
- `USER_MANUAL.txt`
- `COMPLETE_DOCUMENTATION.txt`

## Troubleshooting

- MongoDB connection issues:
  - ensure MongoDB is running
  - or set `MONGODB_URI`

- Ollama-backed chat / summary / RAG not responding:
  - verify Ollama is running
  - verify `OLLAMA_URL`
  - verify the configured model exists locally

- NLLB model not found:
  - set `NLLB_MODEL_PATH`
  - or place the local model in one of the expected directories

- Diarization model not loading:
  - set `HUGGINGFACE_TOKEN`

- Live transcription not working:
  - allow microphone permission in the browser
  - check that the browser supports `MediaRecorder`

- Scanned PDF returns no text:
  - enable OCR with `DOCUMENT_OCR=1`
  - install `pdf2image`, `pytesseract`, and system Tesseract

- NeMo VAD error about tensor dimensions:
  - ensure audio is mono 16k; the pipeline now forces mono conversion
