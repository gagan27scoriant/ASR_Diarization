import json
import os
from datetime import datetime

import ffmpeg
from werkzeug.utils import secure_filename

from app.asr import transcribe, transcribe_live_chunk
from app.config import (
    AUDIO_FOLDER,
    DOCUMENT_FOLDER,
    DOCUMENT_FORMAT_ERROR,
    MEDIA_FORMAT_ERROR,
    RECORDINGS_FOLDER,
    VIDEO_FOLDER,
    is_supported_document,
    is_supported_media,
    is_supported_video,
)
from app.diarization import diarize
from app.history_store import history_json_path, read_history_item, write_history_item
from app.mapper import map_speakers
from app.media_utils import (
    ensure_audio_in_workspace,
    ensure_video_in_workspace,
    ensure_wav,
    enhance_audio_with_demucs,
    extract_audio_from_video,
    extract_text_from_document,
    resolve_media_source,
)
from app.summarize import summarize_text


def resolve_uploaded_or_path_media(uploaded_file, payload) -> tuple[str, str]:
    if uploaded_file and uploaded_file.filename:
        filename = secure_filename(uploaded_file.filename.strip())
        if not filename:
            raise ValueError("Invalid uploaded filename")
        if not is_supported_media(filename):
            raise ValueError(MEDIA_FORMAT_ERROR)

        upload_folder = VIDEO_FOLDER if is_supported_video(filename) else AUDIO_FOLDER
        source_path = os.path.join(upload_folder, filename)
        uploaded_file.save(source_path)
        return source_path, filename

    if not payload:
        raise ValueError("Upload a media file or provide file path")

    incoming_value = (payload.get("file_path") or payload.get("filename") or "").strip()
    if not incoming_value:
        raise ValueError("File path is missing")
    return resolve_media_source(incoming_value)


def process_media_pipeline(source_path: str, filename: str, asr_model, diarization_pipeline) -> dict:
    source_video = ""
    if is_supported_video(filename):
        source_path, source_video = ensure_video_in_workspace(source_path, filename)
        audio_path, extracted_audio_filename = extract_audio_from_video(source_path, source_video)
        audio_path, processed_filename = ensure_wav(audio_path, extracted_audio_filename)
    else:
        audio_path = source_path
        processed_filename = filename

    audio_path, processed_filename = ensure_wav(audio_path, processed_filename)
    audio_path, processed_filename = ensure_audio_in_workspace(audio_path, processed_filename)
    before_audio_path = audio_path
    before_audio_filename = processed_filename

    try:
        audio_path, processed_filename = enhance_audio_with_demucs(audio_path, processed_filename)
    except Exception as demucs_err:
        print(f"⚠️ Demucs skipped, using original audio: {demucs_err}")
        audio_path, processed_filename = before_audio_path, before_audio_filename

    print("\n" + "=" * 50)
    print("Processing:", filename)
    if source_video:
        print("Video source:", source_video)
    print("Before audio:", before_audio_filename)
    print("After audio:", processed_filename)
    if processed_filename != filename:
        print("Converted to:", processed_filename)

    print("→ Running transcription...")
    transcription_segments = transcribe(asr_model, audio_path)

    print("→ Running speaker diarization...")
    diarization_result = diarize(diarization_pipeline, audio_path)

    print("→ Mapping speakers...")
    final_output = map_speakers(transcription_segments, diarization_result)

    asr_seconds = 0.0
    for seg in transcription_segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        if end > start:
            asr_seconds += end - start

    mapped_seconds = 0.0
    for seg in final_output:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        if end > start:
            mapped_seconds += end - start

    if transcription_segments and (not final_output or (asr_seconds > 0 and (mapped_seconds / asr_seconds) < 0.8)):
        print(
            "⚠️ Mapping under-coverage detected. Falling back to ASR-only timeline "
            f"(mapped={mapped_seconds:.2f}s, asr={asr_seconds:.2f}s)."
        )
        final_output = [
            {
                "speaker": "SPEAKER_00",
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": (seg.get("text") or "").strip(),
            }
            for seg in transcription_segments
            if (seg.get("text") or "").strip()
        ]

    output_stem = secure_filename(os.path.splitext(processed_filename)[0]) or "output"
    session_id = f"{output_stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_file = history_json_path(session_id)

    payload = {
        "session_id": session_id,
        "title": filename,
        "processed_file": processed_filename,
        "before_audio_file": before_audio_filename,
        "after_audio_file": processed_filename,
        "source_video": source_video,
        "transcript": final_output,
        "summary": "",
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

    print("✅ Completed:", processed_filename)
    return {
        "session_id": session_id,
        "transcript": final_output,
        "summary": "",
        "processed_file": processed_filename,
        "before_audio_file": before_audio_filename,
        "after_audio_file": processed_filename,
        "source_video": source_video,
    }


def transcribe_live_audio_chunk(uploaded_chunk, asr_model) -> str:
    safe_name = secure_filename(uploaded_chunk.filename) or "chunk.webm"
    chunk_ext = os.path.splitext(safe_name)[1].lower() or ".webm"
    chunk_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    raw_path = os.path.join(RECORDINGS_FOLDER, f"chunk_{chunk_id}{chunk_ext}")
    wav_path = os.path.join(RECORDINGS_FOLDER, f"chunk_{chunk_id}.wav")

    uploaded_chunk.save(raw_path)

    try:
        (
            ffmpeg
            .input(raw_path)
            .output(wav_path, acodec="pcm_s16le", ac=1, ar=16000)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        details = e.stderr.decode("utf-8", errors="ignore") if e.stderr else str(e)
        raise RuntimeError(f"Chunk conversion failed: {details}")

    try:
        return transcribe_live_chunk(asr_model, wav_path)
    finally:
        try:
            if os.path.isfile(raw_path):
                os.remove(raw_path)
            if os.path.isfile(wav_path):
                os.remove(wav_path)
        except Exception:
            pass


def summarize_and_persist(content: str, meeting_title: str, meeting_date: str, meeting_place: str, session_id: str) -> str:
    summary = summarize_text(content, meeting_title, meeting_date, meeting_place)
    if session_id:
        history_data = read_history_item(session_id)
        if history_data is not None:
            history_data["summary"] = summary
            if not write_history_item(session_id, history_data):
                print(f"⚠️ Failed to persist summary to history '{session_id}'")
    return summary


def process_document_upload(uploaded_file, meeting_title: str, meeting_date: str, meeting_place: str) -> dict:
    filename = secure_filename(uploaded_file.filename.strip())
    if not filename:
        raise ValueError("Invalid document filename")
    if not is_supported_document(filename):
        raise ValueError(DOCUMENT_FORMAT_ERROR)

    doc_path = os.path.join(DOCUMENT_FOLDER, filename)
    uploaded_file.save(doc_path)

    extracted_text = extract_text_from_document(doc_path, filename)
    if not extracted_text:
        raise ValueError("No readable text found in document")

    summary = summarize_text(extracted_text, meeting_title, meeting_date, meeting_place)
    return {
        "summary": summary,
        "document_filename": filename,
        "document_type": os.path.splitext(filename.lower())[1].lstrip("."),
        "document_text": extracted_text,
    }
