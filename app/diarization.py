import os
import torch
from pyannote.audio import Pipeline


def load_diarization():
    token = os.getenv("HUGGINGFACE_TOKEN")
    preferred_model = os.getenv("DIARIZATION_MODEL_ID", "pyannote/speaker-diarization-community-1").strip()
    fallback_models = [
        m.strip()
        for m in os.getenv("DIARIZATION_FALLBACK_MODELS", "pyannote/speaker-diarization-3.1").split(",")
        if m.strip()
    ]

    last_error = None
    pipeline = None
    for model_id in [preferred_model] + fallback_models:
        try:
            print(f"Loading diarization model: {model_id}")
            pipeline = Pipeline.from_pretrained(model_id, token=token)
            print(f"Diarization model loaded: {model_id}")
            break
        except Exception as e:
            last_error = e
            print(f"Failed loading diarization model '{model_id}': {e}")

    if pipeline is None:
        raise RuntimeError(f"Unable to load any diarization model: {last_error}")

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    return pipeline



def diarize(pipeline, audio_path):
    min_speakers = os.getenv("DIARIZATION_MIN_SPEAKERS")
    max_speakers = os.getenv("DIARIZATION_MAX_SPEAKERS")

    kwargs = {}
    if min_speakers and min_speakers.isdigit():
        kwargs["min_speakers"] = int(min_speakers)
    if max_speakers and max_speakers.isdigit():
        kwargs["max_speakers"] = int(max_speakers)
    if "min_speakers" in kwargs and "max_speakers" in kwargs and kwargs["min_speakers"] > kwargs["max_speakers"]:
        kwargs["min_speakers"], kwargs["max_speakers"] = kwargs["max_speakers"], kwargs["min_speakers"]

    return pipeline(audio_path, **kwargs)
