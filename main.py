import os
import warnings

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from app.asr import load_asr
from app.config import (
    AUDIO_FOLDER,
    DOCUMENT_FOLDER,
    DOCUMENT_FORMAT_ERROR,
    VIDEO_FOLDER,
    ensure_workspace_folders,
    is_supported_document,
)
from app.diarization import load_diarization
from app.history_store import (
    delete_history_item as remove_history_item,
    history_json_path,
    list_history_entries,
    read_history_item,
    rename_history_item as rename_history_record,
    update_history_transcript,
)
from app.processing_service import (
    process_document_upload,
    process_media_pipeline,
    resolve_uploaded_or_path_media,
    summarize_and_persist,
    transcribe_live_audio_chunk,
)
from app.translation import get_translator


warnings.filterwarnings(
    "ignore",
    message=r"(?s).*torchcodec is not installed correctly.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*torchaudio\._backend\.list_audio_backends has been deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*implementation will be changed to use torchaudio\.load_with_torchcodec.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*StreamingMediaDecoder has been deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*Module 'speechbrain\.pretrained' was deprecated.*",
    category=UserWarning,
)


app = Flask(__name__, template_folder="templates", static_folder="static")
ensure_workspace_folders()


print("🚀 Loading ASR model...")
preferred_asr = os.getenv("ASR_MODEL_SIZE", "medium")
try:
    asr_model = load_asr(preferred_asr)
except Exception as asr_err:
    print(f"⚠️ Failed to load ASR model '{preferred_asr}': {asr_err}")
    print("↪ Falling back to ASR model 'medium'...")
    asr_model = load_asr("medium")

print("🚀 Loading diarization model...")
diarization_pipeline = load_diarization()
print("✅ Models ready\n")


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


@app.route("/process", methods=["POST"])
def process_audio():
    try:
        uploaded_file = request.files.get("audio_file")
        payload = request.get_json(silent=True) if not uploaded_file else None
        source_path, filename = resolve_uploaded_or_path_media(uploaded_file, payload)
        result = process_media_pipeline(source_path, filename, asr_model, diarization_pipeline)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/transcribe_chunk", methods=["POST"])
def transcribe_chunk():
    try:
        uploaded_chunk = request.files.get("audio_chunk")
        if not uploaded_chunk or not uploaded_chunk.filename:
            return jsonify({"error": "Audio chunk missing"}), 400
        text = transcribe_live_audio_chunk(uploaded_chunk, asr_model)
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/summarize_text", methods=["POST"])
def summarize_from_text():
    try:
        data = request.get_json() or {}
        text = data.get("content", "")
        meeting_title = (data.get("meeting_title") or "").strip()
        meeting_date = (data.get("meeting_date") or "").strip()
        meeting_place = (data.get("meeting_place") or "").strip()

        if not text or not text.strip():
            return jsonify({"error": "Content missing"}), 400
        if not meeting_title or not meeting_date or not meeting_place:
            return jsonify({"error": "Meeting title, date, and place are required"}), 400

        print("→ Generating summary from exported text...")
        summary = summarize_and_persist(
            text,
            meeting_title,
            meeting_date,
            meeting_place,
            (data.get("session_id") or "").strip(),
        )
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
        if not is_supported_document(uploaded_file.filename):
            return jsonify({"error": DOCUMENT_FORMAT_ERROR}), 400

        meeting_title = (request.form.get("meeting_title") or "").strip()
        meeting_date = (request.form.get("meeting_date") or "").strip()
        meeting_place = (request.form.get("meeting_place") or "").strip()
        if not meeting_title or not meeting_date or not meeting_place:
            return jsonify({"error": "Meeting title, date, and place are required"}), 400

        print("→ Generating summary from uploaded document...")
        result = process_document_upload(
            uploaded_file,
            meeting_title=meeting_title,
            meeting_date=meeting_date,
            meeting_place=meeting_place,
        )
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print("❌ Document Processing Error:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/translate", methods=["POST"])
def translate_content():
    try:
        data = request.get_json(silent=True) or {}
        target_lang = (data.get("target_lang") or "").strip()
        source_lang = (data.get("source_lang") or "").strip()
        text = data.get("text")
        texts = data.get("texts")

        if not target_lang:
            return jsonify({"error": "target_lang is required"}), 400

        translator = get_translator()
        resolved_target = translator.resolve_lang_code(target_lang, is_target=True)
        resolved_source = translator.resolve_lang_code(source_lang or "", is_target=False)

        if texts is not None:
            if not isinstance(texts, list):
                return jsonify({"error": "texts must be an array"}), 400
            translated_texts = translator.translate_lines(
                [str(x or "") for x in texts],
                target_lang=resolved_target,
                source_lang=resolved_source,
            )
            return jsonify(
                {
                    "texts": translated_texts,
                    "target_lang": resolved_target,
                    "source_lang": resolved_source,
                }
            )

        if not isinstance(text, str):
            return jsonify({"error": "text must be a string"}), 400

        translated_text = translator.translate_text(
            text,
            target_lang=resolved_target,
            source_lang=resolved_source,
        )
        return jsonify(
            {
                "text": translated_text,
                "target_lang": resolved_target,
                "source_lang": resolved_source,
            }
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history", methods=["GET"])
def list_history():
    try:
        return jsonify({"history": list_history_entries()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history/<session_id>", methods=["GET"])
def get_history_item(session_id):
    try:
        data = read_history_item(session_id)
        if data is None:
            return jsonify({"error": "History not found"}), 404

        return jsonify(
            {
                "session_id": session_id,
                "title": data.get("title") or session_id,
                "processed_file": data.get("processed_file") or "",
                "before_audio_file": data.get("before_audio_file") or data.get("processed_file") or "",
                "after_audio_file": data.get("after_audio_file") or data.get("processed_file") or "",
                "source_video": data.get("source_video") or "",
                "transcript": data.get("transcript") or [],
                "summary": data.get("summary") or "",
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history/<session_id>/transcript", methods=["POST"])
def save_history_transcript(session_id):
    try:
        if not history_json_path(session_id) or read_history_item(session_id) is None:
            return jsonify({"error": "History not found"}), 404

        payload = request.get_json(silent=True) or {}
        transcript = payload.get("transcript")
        summary = payload.get("summary")

        if transcript is not None and not isinstance(transcript, list):
            return jsonify({"error": "Invalid transcript payload"}), 400
        if summary is not None and not isinstance(summary, str):
            return jsonify({"error": "Invalid summary payload"}), 400

        update_history_transcript(session_id, transcript=transcript, summary=summary)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history/<session_id>", methods=["PATCH"])
def rename_history_item(session_id):
    try:
        if not history_json_path(session_id) or read_history_item(session_id) is None:
            return jsonify({"error": "History not found"}), 404

        payload = request.get_json(silent=True) or {}
        new_title = (payload.get("title") or "").strip()
        if not new_title:
            return jsonify({"error": "Title is required"}), 400

        rename_history_record(session_id, new_title)
        return jsonify({"ok": True, "title": new_title})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history/<session_id>", methods=["DELETE"])
def delete_history_item(session_id):
    try:
        if not remove_history_item(session_id):
            return jsonify({"error": "History not found"}), 404
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "1").strip().lower() in {"1", "true", "yes", "on"}
    use_reloader = os.getenv("FLASK_USE_RELOADER", "0").strip().lower() in {"1", "true", "yes", "on"}
    app.run(host="0.0.0.0", port=5000, debug=debug_mode, use_reloader=use_reloader)
