import os

import torch
from faster_whisper import WhisperModel


def load_asr(model_size=None):
    model_path = os.getenv("ASR_MODEL_PATH", "").strip()
    model_size = model_size or os.getenv("ASR_MODEL_SIZE", "large-v3")
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    compute_type = "float16" if use_cuda else "int8"
    model_source = model_path if model_path else model_size

    print(f"Loading ASR model '{model_source}' on {device.upper()}...")

    model = WhisperModel(
        model_source,
        device=device,
        compute_type=compute_type
    )

    print(f"ASR model loaded on {device.upper()} ({compute_type})")
    return model


def transcribe(model, audio_path):
    """
    Transcribe audio and return Whisper-like output.
    """

    segments, _ = model.transcribe(
        audio_path,
        beam_size=6,
        best_of=6,
        temperature=0.0,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 450},
        task="translate",  # convert transcript content to English
        language=None,
        condition_on_previous_text=True,
        word_timestamps=True,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0
    )

    result = []
    for seg in segments:
        result.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text
        })

    return result
