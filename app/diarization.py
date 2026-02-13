import os
import torch
from pyannote.audio import Pipeline


def load_diarization():
    token = os.getenv("HUGGINGFACE_TOKEN")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=token   # ✅ correct argument now
    )

    return pipeline



def diarize(pipeline, audio_path):
    return pipeline(audio_path)
