import os
import shutil
import subprocess
import sys
from glob import glob

import ffmpeg
from docx import Document as DocxDocument
from moviepy import VideoFileClip
from pypdf import PdfReader
from werkzeug.utils import secure_filename

from app.config import (
    AUDIO_FOLDER,
    DOCUMENT_FORMAT_ERROR,
    MEDIA_FORMAT_ERROR,
    SUPPORTED_DOCUMENT_EXTENSIONS,
    VIDEO_FOLDER,
    is_supported_media,
)


def ensure_wav(audio_path: str, filename: str) -> tuple[str, str]:
    ext = os.path.splitext(filename.lower())[1]
    if ext == ".wav":
        return audio_path, filename

    safe_base = secure_filename(os.path.splitext(filename)[0]) or "uploaded_audio"
    wav_filename = f"{safe_base}.wav"
    wav_path = os.path.join(AUDIO_FOLDER, wav_filename)

    try:
        (
            ffmpeg
            .input(audio_path)
            .output(wav_path, acodec="pcm_s16le", ac=1, ar=16000)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        details = e.stderr.decode("utf-8", errors="ignore") if e.stderr else str(e)
        raise RuntimeError(f"Audio conversion failed: {details}")

    return wav_path, wav_filename


def resolve_media_source(path_or_name: str) -> tuple[str, str]:
    value = (path_or_name or "").strip()
    if not value:
        raise ValueError("File path is missing")

    expanded = os.path.expanduser(value)
    if os.path.isfile(expanded):
        filename = os.path.basename(expanded)
        if not is_supported_media(filename):
            raise ValueError(MEDIA_FORMAT_ERROR)
        return expanded, filename

    filename = os.path.basename(expanded)
    if not filename:
        raise ValueError("Invalid file path")
    if not is_supported_media(filename):
        raise ValueError(MEDIA_FORMAT_ERROR)

    fallback_audio_path = os.path.join(AUDIO_FOLDER, filename)
    fallback_video_path = os.path.join(VIDEO_FOLDER, filename)
    if os.path.isfile(fallback_audio_path):
        return fallback_audio_path, filename
    if os.path.isfile(fallback_video_path):
        return fallback_video_path, filename

    raise FileNotFoundError(
        "Media file not found. Provide a valid file path or upload from UI."
    )


def ensure_audio_in_workspace(audio_path: str, filename: str) -> tuple[str, str]:
    target_filename = secure_filename(filename) or "audio.wav"
    target_path = os.path.join(AUDIO_FOLDER, target_filename)

    if os.path.abspath(audio_path) == os.path.abspath(target_path):
        return audio_path, target_filename

    shutil.copy2(audio_path, target_path)
    return target_path, target_filename


def ensure_video_in_workspace(video_path: str, filename: str) -> tuple[str, str]:
    target_filename = secure_filename(filename) or "video.mp4"
    target_path = os.path.join(VIDEO_FOLDER, target_filename)

    if os.path.abspath(video_path) == os.path.abspath(target_path):
        return video_path, target_filename

    shutil.copy2(video_path, target_path)
    return target_path, target_filename


def extract_audio_from_video(video_path: str, filename: str) -> tuple[str, str]:
    safe_base = secure_filename(os.path.splitext(filename)[0]) or "uploaded_video"
    wav_filename = f"{safe_base}_from_video.wav"
    wav_path = os.path.join(AUDIO_FOLDER, wav_filename)

    try:
        with VideoFileClip(video_path) as clip:
            if clip.audio is None:
                raise RuntimeError("Video has no audio track")
            clip.audio.write_audiofile(
                wav_path,
                codec="pcm_s16le",
                fps=16000,
                ffmpeg_params=["-ac", "1"],
                logger=None
            )
    except Exception as e:
        raise RuntimeError(f"Video audio extraction failed: {e}")

    return wav_path, wav_filename


def enhance_audio_with_demucs(audio_path: str, filename: str) -> tuple[str, str]:
    enabled = (os.getenv("DEMUCS_ENABLED", "1") or "").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return audio_path, filename

    model_name = (os.getenv("DEMUCS_MODEL", "htdemucs") or "").strip() or "htdemucs"
    output_root = (os.getenv("DEMUCS_OUTPUT_FOLDER", "demucs_outputs") or "").strip() or "demucs_outputs"
    two_stems = (os.getenv("DEMUCS_TWO_STEMS", "vocals") or "").strip() or "vocals"
    safe_base = secure_filename(os.path.splitext(filename)[0]) or "audio"

    cmd = [
        sys.executable,
        "-m",
        "demucs.separate",
        "-n",
        model_name,
        "--two-stems",
        two_stems,
        "-o",
        output_root,
        audio_path,
    ]

    demucs_device = (os.getenv("DEMUCS_DEVICE", "") or "").strip().lower()
    if demucs_device in {"cpu", "cuda"}:
        cmd.extend(["--device", demucs_device])

    print(f"→ Running Demucs ({model_name}, stem={two_stems})...")
    run_result = subprocess.run(cmd, capture_output=True, text=True)
    if run_result.returncode != 0:
        stderr = (run_result.stderr or "").strip()
        stdout = (run_result.stdout or "").strip()
        details = stderr[-600:] or stdout[-600:] or "Unknown Demucs error"
        raise RuntimeError(f"Demucs separation failed: {details}")

    track_base = os.path.splitext(os.path.basename(audio_path))[0]
    stem_glob = os.path.join(output_root, model_name, track_base, f"{two_stems}.*")
    stem_files = sorted(glob(stem_glob))
    if not stem_files:
        raise RuntimeError("Demucs output stem not found")

    enhanced_src = stem_files[0]
    enhanced_filename = f"{safe_base}_demucs.wav"
    enhanced_path = os.path.join(AUDIO_FOLDER, enhanced_filename)

    try:
        (
            ffmpeg
            .input(enhanced_src)
            .output(enhanced_path, acodec="pcm_s16le", ac=1, ar=16000)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        details = e.stderr.decode("utf-8", errors="ignore") if e.stderr else str(e)
        raise RuntimeError(f"Demucs output conversion failed: {details}")

    return enhanced_path, enhanced_filename


def extract_text_from_document(file_path: str, filename: str) -> str:
    ext = os.path.splitext(filename.lower())[1]

    if ext == ".pdf":
        reader = PdfReader(file_path)
        text_parts = []
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts).strip()

    if ext == ".docx":
        doc = DocxDocument(file_path)
        parts = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        return "\n".join(parts).strip()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()

    if ext not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise ValueError(DOCUMENT_FORMAT_ERROR)
    raise ValueError(DOCUMENT_FORMAT_ERROR)
