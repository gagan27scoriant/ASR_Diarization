# Overview
AI Knowledge Studio (ASR Studio) is an on-premise speech and document intelligence platform that transforms raw meeting media into structured outputs for review, search, and reporting. The application supports audio files, video files, and document uploads, then routes each request through a query-first agent workflow to produce transcripts, speaker-separated conversations, summaries, Q&A responses, translations, and export-ready meeting artifacts. The platform is designed for local deployment with enterprise-style data control: processing, indexing, and history management run inside the host environment, and core pipelines rely on self-hosted or locally available model services.

The system combines Faster Whisper for speech recognition, Pyannote (default) or NeMo (optional) for diarization, Ollama-hosted LLMs for summarization and retrieval responses, sentence-transformer embeddings for semantic retrieval, and NLLB-based multilingual translation. Together, these components provide an end-to-end pipeline that converts unstructured media and documents into searchable, contextual, and operationally useful outputs.

## End-to-End Processing Flow
The primary workflow begins when a user uploads a media or document file and submits a natural-language instruction through the agent bar. The backend determines intent and content type, then triggers the appropriate tool path.

For audio and video:
- Media is normalized through FFmpeg (including audio extraction from video when required).
- Audio is converted to mono 16 kHz for robust downstream processing.
- Faster Whisper transcribes speech to English text output with timestamps.
- Speaker diarization assigns speaker segments using Pyannote (or NeMo if configured).
- Speaker mapping aligns transcription segments with diarization timelines.
- Transcript sessions are saved with metadata and can immediately be used for summary, Q&A, semantic search, translation, and export.

For documents (PDF/DOCX/TXT/image):
- Text is extracted using file-type-specific extraction logic.
- Optional OCR is available for scanned content.
- Extracted text is chunked and embedded for retrieval.
- Document summaries are generated using configured summarization backend.
- Document chunks and metadata are stored for history-aware document Q&A.

## Agent-Oriented Interaction Model
The platform runs in two operating modes:

1. Plain Chat Mode  
When no transcript or document context is active, the assistant behaves like a normal conversational chatbot with short-term session memory.

2. Contextual Agent Mode  
When a transcript or document context is active (or a file is uploaded), the system routes to task-specific tools such as:
- media processing
- transcript Q&A
- document Q&A
- transcript summarization
- translation
- semantic search
- text-to-speech

This architecture keeps user interaction natural-language-first while preserving deterministic backend tool execution and permission-aware API behavior.

## Summarization, Minutes, and Retrieval
After transcription or document ingestion, users can generate structured outputs including:
- concise narrative summaries
- minutes-style summaries for meeting documentation
- contextual answers grounded in transcript/document content

Retrieval-based Q&A is backed by stored transcript/document chunks and embeddings, allowing follow-up questions to use semantic context rather than a single-turn prompt. Transcript and document history are persisted and can be reopened from sidebar history views, enabling iterative analysis without reprocessing source files.

## Multilingual and Speech Output
The system supports translation workflows through local/offline NLLB model paths and can translate selected outputs such as transcript responses and summaries. For spoken playback, gTTS-based generation is available, with browser speech fallback behavior in UI paths when needed.

## Desktop and Preview Experience
The application can run in browser mode or as a Tauri desktop shell. In desktop mode, the backend is started locally and the UI is rendered inside a native window. For PDF preview consistency across browser and Tauri environments, inline rendering uses a PDF.js-based viewer path for sidebar mini-preview and full panel preview, avoiding dependency on native webview PDF plugins.

## Security, Data Locality, and Operations
The platform includes authentication, role/permission policies, audit logging, and department-aware visibility controls for history/documents. All generated artifacts (transcripts, summaries, chunks, history metadata) are stored locally in project-managed folders and database collections, supporting secure internal deployment patterns. Because model backends (ASR, diarization, embedding, and LLM orchestration) are hosted locally or within controlled infrastructure, organizations can operate the system with strong privacy guarantees and minimal external data exposure.

## Result
ASR Studio delivers a complete pipeline from ingestion to insight: users move from raw meeting recordings and source documents to speaker-aware transcripts, contextual Q&A, summary intelligence, multilingual output, and exportable records in a single integrated application flow.
