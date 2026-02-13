from flask import Flask, request, jsonify, render_template
import os
import json

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


# ----------------------------
# Serve Frontend
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ----------------------------
# Process Audio API
# Called from index.html fetch("/process")
# ----------------------------
@app.route("/process", methods=["POST"])
def process_audio():

    try:
        data = request.get_json()

        if not data or "filename" not in data:
            return jsonify({"error": "Filename missing"}), 400

        filename = data["filename"].strip()

        # Prevent path injection
        filename = os.path.basename(filename)

        audio_path = os.path.join(AUDIO_FOLDER, filename)

        if not os.path.exists(audio_path):
            return jsonify({"error": "Audio file not found"}), 404

        print("\n" + "="*50)
        print("Processing:", filename)

        # ----------------------------
        # ASR
        # ----------------------------
        print("→ Running transcription...")
        transcription = transcribe(asr_model, audio_path)

        # ----------------------------
        # Diarization
        # ----------------------------
        print("→ Running speaker diarization...")
        diarization_result = diarize(pipeline, audio_path)

        # ----------------------------
        # Speaker Mapping
        # ----------------------------
        print("→ Mapping speakers...")
        final_output = map_speakers(transcription, diarization_result)

        # ----------------------------
        # Summary
        # ----------------------------
        print("→ Generating summary...")
        full_text = " ".join([seg["text"] for seg in final_output])
        summary = summarize_text(full_text)

        # ----------------------------
        # Save JSON Output
        # ----------------------------
        output_file = os.path.join(
            OUTPUT_FOLDER,
            f"{os.path.splitext(filename)[0]}.json"
        )

        with open(output_file, "w") as f:
            json.dump(
                {
                    "transcript": final_output,
                    "summary": summary
                },
                f,
                indent=4
            )

        print("✅ Completed:", filename)

        # IMPORTANT → return exactly what UI expects
        return jsonify({
            "transcript": final_output,
            "summary": summary
        })

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500


# ----------------------------
# Run Server
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
