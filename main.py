from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import json
import shutil

import ffmpeg
from docx import Document as DocxDocument
from moviepy import VideoFileClip
from pypdf import PdfReader
from werkzeug.utils import secure_filename

from app.asr import load_asr, transcribe
from app.diarization import load_diarization, diarize
from app.mapper import map_speakers
from app.summarize import summarize_text


# ----------------------------
# Flask Setup
# ----------------------------
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

AUDIO_FOLDER = "audio"
VIDEO_FOLDER = "videos"
OUTPUT_FOLDER = "outputs"
DOCUMENT_FOLDER = "documents"

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".aac", ".aiff", ".wma", ".amr", ".opus"
}
SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".mpeg", ".3gp"
}
SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".txt"}

os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(DOCUMENT_FOLDER, exist_ok=True)


# ----------------------------
# Load models ONCE (important)
# ----------------------------
print("🚀 Loading ASR model...")
preferred_asr = os.getenv("ASR_MODEL_SIZE", "large-v3")
try:
    asr_model = load_asr(preferred_asr)
except Exception as asr_err:
    print(f"⚠️ Failed to load ASR model '{preferred_asr}': {asr_err}")
    print("↪ Falling back to ASR model 'medium'...")
    asr_model = load_asr("medium")

print("🚀 Loading diarization model...")
pipeline = load_diarization()

print("✅ Models ready\n")


def _is_supported_audio(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in SUPPORTED_AUDIO_EXTENSIONS


def _is_supported_video(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in SUPPORTED_VIDEO_EXTENSIONS


def _is_supported_media(filename: str) -> bool:
    return _is_supported_audio(filename) or _is_supported_video(filename)


def _ensure_wav(audio_path: str, filename: str) -> tuple[str, str]:
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


def _resolve_media_source(path_or_name: str) -> tuple[str, str]:
    value = (path_or_name or "").strip()
    if not value:
        raise ValueError("File path is missing")

    expanded = os.path.expanduser(value)
    if os.path.isfile(expanded):
        filename = os.path.basename(expanded)
        if not _is_supported_media(filename):
            raise ValueError(
                "Unsupported media format. Audio: .wav, .mp3, .aac, .aiff, .wma, .amr, .opus | "
                "Video: .mp4, .mkv, .avi, .mov, .wmv, .mpeg, .3gp"
            )
        return expanded, filename

    filename = os.path.basename(expanded)
    if not filename:
        raise ValueError("Invalid file path")
    if not _is_supported_media(filename):
        raise ValueError(
            "Unsupported media format. Audio: .wav, .mp3, .aac, .aiff, .wma, .amr, .opus | "
            "Video: .mp4, .mkv, .avi, .mov, .wmv, .mpeg, .3gp"
        )

    fallback_audio_path = os.path.join(AUDIO_FOLDER, filename)
    fallback_video_path = os.path.join(VIDEO_FOLDER, filename)
    if os.path.isfile(fallback_audio_path):
        return fallback_audio_path, filename
    if os.path.isfile(fallback_video_path):
        return fallback_video_path, filename

    raise FileNotFoundError(
        "Media file not found. Provide a valid file path or upload from UI."
    )


def _ensure_audio_in_workspace(audio_path: str, filename: str) -> tuple[str, str]:
    target_filename = secure_filename(filename) or "audio.wav"
    target_path = os.path.join(AUDIO_FOLDER, target_filename)

    if os.path.abspath(audio_path) == os.path.abspath(target_path):
        return audio_path, target_filename

    shutil.copy2(audio_path, target_path)
    return target_path, target_filename


def _ensure_video_in_workspace(video_path: str, filename: str) -> tuple[str, str]:
    target_filename = secure_filename(filename) or "video.mp4"
    target_path = os.path.join(VIDEO_FOLDER, target_filename)

    if os.path.abspath(video_path) == os.path.abspath(target_path):
        return video_path, target_filename

    shutil.copy2(video_path, target_path)
    return target_path, target_filename


def _extract_audio_from_video(video_path: str, filename: str) -> tuple[str, str]:
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


def _extract_text_from_document(file_path: str, filename: str) -> str:
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

    raise ValueError("Unsupported document format. Use: .pdf, .docx, .txt")


# ----------------------------
# Serve Frontend
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/audio/<path:filename>")
def serve_audio(filename):
    safe_name = secure_filename(filename)
    if not safe_name:
        return jsonify({"error": "Invalid audio filename"}), 400
    return send_from_directory(AUDIO_FOLDER, safe_name)


@app.route("/videos/<path:filename>")
def serve_video(filename):
    safe_name = secure_filename(filename)
    if not safe_name:
        return jsonify({"error": "Invalid video filename"}), 400
    return send_from_directory(VIDEO_FOLDER, safe_name)


@app.route("/documents/<path:filename>")
def serve_document(filename):
    safe_name = secure_filename(filename)
    if not safe_name:
        return jsonify({"error": "Invalid document filename"}), 400
    return send_from_directory(DOCUMENT_FOLDER, safe_name)


# ----------------------------
# Process Audio API
# Called from index.html fetch("/process")
# ----------------------------
@app.route("/process", methods=["POST"])
def process_audio():

    try:
        filename = ""
        source_path = ""
        source_video = ""

        uploaded_file = request.files.get("audio_file")
        if uploaded_file and uploaded_file.filename:
            filename = secure_filename(uploaded_file.filename.strip())
            if not filename:
                return jsonify({"error": "Invalid uploaded filename"}), 400

            if not _is_supported_media(filename):
                return jsonify({
                    "error": "Unsupported media format. Audio: .wav, .mp3, .aac, .aiff, .wma, .amr, .opus | "
                             "Video: .mp4, .mkv, .avi, .mov, .wmv, .mpeg, .3gp"
                }), 400

            upload_folder = VIDEO_FOLDER if _is_supported_video(filename) else AUDIO_FOLDER
            source_path = os.path.join(upload_folder, filename)
            uploaded_file.save(source_path)
        else:
            data = request.get_json(silent=True)
            if not data:
                return jsonify({"error": "Upload a media file or provide file path"}), 400

            incoming_value = (data.get("file_path") or data.get("filename") or "").strip()
            if not incoming_value:
                return jsonify({"error": "File path is missing"}), 400

            try:
                source_path, filename = _resolve_media_source(incoming_value)
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
            except FileNotFoundError as e:
                return jsonify({"error": str(e)}), 404

        if _is_supported_video(filename):
            source_path, source_video = _ensure_video_in_workspace(source_path, filename)
            audio_path, extracted_audio_filename = _extract_audio_from_video(source_path, source_video)
            audio_path, processed_filename = _ensure_wav(audio_path, extracted_audio_filename)
        else:
            audio_path = source_path
            processed_filename = filename
            source_video = ""

        # Convert anything except wav into wav before pipeline
        audio_path, processed_filename = _ensure_wav(audio_path, processed_filename)
        audio_path, processed_filename = _ensure_audio_in_workspace(audio_path, processed_filename)

        print("\n" + "="*50)
        print("Processing:", filename)
        if source_video:
            print("Video source:", source_video)
        if processed_filename != filename:
            print("Converted to:", processed_filename)

        # ----------------------------
        # ASR (Faster-Whisper)
        # ----------------------------
        print("→ Running transcription...")
        # Faster-Whisper returns a list of segments
        transcription_segments = transcribe(asr_model, audio_path)

        # ----------------------------
        # Diarization
        # ----------------------------
        print("→ Running speaker diarization...")
        diarization_result = diarize(pipeline, audio_path)

        # ----------------------------
        # Speaker Mapping
        # ----------------------------
        print("→ Mapping speakers...")
        final_output = map_speakers(transcription_segments, diarization_result)

        # ----------------------------
        # Save JSON Output
        # ----------------------------
        output_stem = secure_filename(os.path.splitext(processed_filename)[0]) or "output"
        output_file = os.path.join(OUTPUT_FOLDER, f"{output_stem}.json")

        with open(output_file, "w") as f:
            json.dump(
                {
                    "transcript": final_output,
                    "summary": ""
                },
                f,
                indent=4
            )

        print("✅ Completed:", processed_filename)

        return jsonify({
            "transcript": final_output,
            "summary": "",
            "processed_file": processed_filename,
            "source_video": source_video
        })

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/summarize_text", methods=["POST"])
def summarize_from_text():
    try:
        data = request.get_json()
        text = (data or {}).get("content", "")
        meeting_title = (data or {}).get("meeting_title", "").strip()
        meeting_date = (data or {}).get("meeting_date", "").strip()
        meeting_place = (data or {}).get("meeting_place", "").strip()

        if not text or not text.strip():
            return jsonify({"error": "Content missing"}), 400
        if not meeting_title or not meeting_date or not meeting_place:
            return jsonify({"error": "Meeting title, date, and place are required"}), 400

        print("→ Generating summary from exported text...")
        summary = summarize_text(text, meeting_title, meeting_date, meeting_place)
        return jsonify({"summary": summary})
    except Exception as e:
        print("❌ Summary Error:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/process_document", methods=["POST"])
def process_document():
    try:
        uploaded_file = request.files.get("document_file")
        if not uploaded_file or not uploaded_file.filename:
            return jsonify({"error": "Document file missing"}), 400

        filename = secure_filename(uploaded_file.filename.strip())
        if not filename:
            return jsonify({"error": "Invalid document filename"}), 400

        ext = os.path.splitext(filename.lower())[1]
        if ext not in SUPPORTED_DOCUMENT_EXTENSIONS:
            return jsonify({"error": "Unsupported document format. Use: .pdf, .docx, .txt"}), 400

        meeting_title = (request.form.get("meeting_title") or "").strip()
        meeting_date = (request.form.get("meeting_date") or "").strip()
        meeting_place = (request.form.get("meeting_place") or "").strip()
        if not meeting_title or not meeting_date or not meeting_place:
            return jsonify({"error": "Meeting title, date, and place are required"}), 400

        doc_path = os.path.join(DOCUMENT_FOLDER, filename)
        uploaded_file.save(doc_path)

        extracted_text = _extract_text_from_document(doc_path, filename)
        if not extracted_text:
            return jsonify({"error": "No readable text found in document"}), 400

        print("→ Generating summary from uploaded document...")
        summary = summarize_text(extracted_text, meeting_title, meeting_date, meeting_place)

        return jsonify({
            "summary": summary,
            "document_filename": filename,
            "document_type": ext.lstrip("."),
            "document_text": extracted_text
        })
    except Exception as e:
        print("❌ Document Processing Error:", e)
        return jsonify({"error": str(e)}), 500


# ----------------------------
# Run Server
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
