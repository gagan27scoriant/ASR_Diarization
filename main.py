from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import json
import shutil

import ffmpeg
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
OUTPUT_FOLDER = "outputs"

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".aac", ".aiff", ".wma", ".amr", ".opus"
}

os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ----------------------------
# Load models ONCE (important)
# ----------------------------
print("🚀 Loading ASR model...")
asr_model = load_asr("base")  # or "tiny" for faster CPU

print("🚀 Loading diarization model...")
pipeline = load_diarization()

print("✅ Models ready\n")


def _is_supported_audio(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in SUPPORTED_AUDIO_EXTENSIONS


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


def _resolve_audio_source(path_or_name: str) -> tuple[str, str]:
    value = (path_or_name or "").strip()
    if not value:
        raise ValueError("File path is missing")

    expanded = os.path.expanduser(value)
    if os.path.isfile(expanded):
        filename = os.path.basename(expanded)
        if not _is_supported_audio(filename):
            raise ValueError(
                "Unsupported audio format. Use: .wav, .mp3, .aac, .aiff, .wma, .amr, .opus"
            )
        return expanded, filename

    filename = os.path.basename(expanded)
    if not filename:
        raise ValueError("Invalid file path")
    if not _is_supported_audio(filename):
        raise ValueError(
            "Unsupported audio format. Use: .wav, .mp3, .aac, .aiff, .wma, .amr, .opus"
        )

    fallback_audio_path = os.path.join(AUDIO_FOLDER, filename)
    if os.path.isfile(fallback_audio_path):
        return fallback_audio_path, filename

    raise FileNotFoundError(
        "Audio file not found. Provide a valid file path or upload from UI."
    )


def _ensure_audio_in_workspace(audio_path: str, filename: str) -> tuple[str, str]:
    target_filename = secure_filename(filename) or "audio.wav"
    target_path = os.path.join(AUDIO_FOLDER, target_filename)

    if os.path.abspath(audio_path) == os.path.abspath(target_path):
        return audio_path, target_filename

    shutil.copy2(audio_path, target_path)
    return target_path, target_filename


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


# ----------------------------
# Process Audio API
# Called from index.html fetch("/process")
# ----------------------------
@app.route("/process", methods=["POST"])
def process_audio():

    try:
        filename = ""
        audio_path = ""

        uploaded_file = request.files.get("audio_file")
        if uploaded_file and uploaded_file.filename:
            filename = secure_filename(uploaded_file.filename.strip())
            if not filename:
                return jsonify({"error": "Invalid uploaded filename"}), 400

            if not _is_supported_audio(filename):
                return jsonify({
                    "error": "Unsupported audio format. Use: .wav, .mp3, .aac, .aiff, .wma, .amr, .opus"
                }), 400

            audio_path = os.path.join(AUDIO_FOLDER, filename)
            uploaded_file.save(audio_path)
        else:
            data = request.get_json(silent=True)
            if not data:
                return jsonify({"error": "Upload an audio file or provide file path"}), 400

            incoming_value = (data.get("file_path") or data.get("filename") or "").strip()
            if not incoming_value:
                return jsonify({"error": "File path is missing"}), 400

            try:
                audio_path, filename = _resolve_audio_source(incoming_value)
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
            except FileNotFoundError as e:
                return jsonify({"error": str(e)}), 404

        # Convert anything except wav into wav before pipeline
        audio_path, processed_filename = _ensure_wav(audio_path, filename)
        audio_path, processed_filename = _ensure_audio_in_workspace(audio_path, processed_filename)

        print("\n" + "="*50)
        print("Processing:", filename)
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
            "processed_file": processed_filename
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


# ----------------------------
# Run Server
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
