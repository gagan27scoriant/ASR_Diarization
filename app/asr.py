from faster_whisper import WhisperModel


def load_asr(model_size="tiny"):
    print("Loading ASR on GPU...")

    model = WhisperModel(
        model_size,
        device="cuda",
        compute_type="float16"
    )

    print("ASR model loaded on CUDA")
    return model


def transcribe(model, audio_path):
    """
    Transcribe audio and return Whisper-like output.
    """

    segments, _ = model.transcribe(
        audio_path,
        beam_size=1   # faster decoding
    )

    result = []
    for seg in segments:
        result.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text
        })

    return result
