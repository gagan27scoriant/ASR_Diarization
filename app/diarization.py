import os
import torch
from pyannote.audio import Pipeline


def load_diarization():
    token = os.getenv("HUGGINGFACE_TOKEN")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=token   # ✅ correct argument now
    )
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

    return pipeline(audio_path, **kwargs)
