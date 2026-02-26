import os
import torch
import torchaudio
from pyannote.audio import Pipeline

PINNED_DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-community-1"


def load_diarization():
    token = os.getenv("HUGGINGFACE_TOKEN")
    model_id = PINNED_DIARIZATION_MODEL_ID
    print(f"Loading diarization model: {model_id}")
    pipeline = Pipeline.from_pretrained(model_id, token=token)
    print(f"Diarization model loaded: {model_id}")

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    return pipeline



def diarize(pipeline, audio_path):
    num_speakers = os.getenv("DIARIZATION_NUM_SPEAKERS")
    min_speakers = os.getenv("DIARIZATION_MIN_SPEAKERS")
    max_speakers = os.getenv("DIARIZATION_MAX_SPEAKERS")

    kwargs = {}
    if num_speakers and num_speakers.isdigit():
        kwargs["num_speakers"] = int(num_speakers)

    if min_speakers and min_speakers.isdigit():
        kwargs["min_speakers"] = int(min_speakers)
    if max_speakers and max_speakers.isdigit():
        kwargs["max_speakers"] = int(max_speakers)
    if "min_speakers" in kwargs and "max_speakers" in kwargs and kwargs["min_speakers"] > kwargs["max_speakers"]:
        kwargs["min_speakers"], kwargs["max_speakers"] = kwargs["max_speakers"], kwargs["min_speakers"]

    preload_audio = os.getenv("DIARIZATION_PRELOAD_AUDIO", "1").strip().lower() in {"1", "true", "yes", "on"}
    if not preload_audio:
        return pipeline(audio_path, **kwargs)

    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        return pipeline({"waveform": waveform, "sample_rate": sample_rate}, **kwargs)
    except Exception as e:
        print(f"⚠️ Preloaded diarization input failed, falling back to file path: {e}")
        return pipeline(audio_path, **kwargs)
