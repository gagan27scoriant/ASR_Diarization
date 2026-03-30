import os
import warnings

import torch
import torchaudio

PINNED_DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-community-1"


def load_diarization():
    # Shim for torchaudio builds that only expose torchcodec APIs.
    if not hasattr(torchaudio, "AudioMetaData"):
        from dataclasses import dataclass

        @dataclass
        class AudioMetaData:  # minimal replacement for pyannote usage
            sample_rate: int
            num_frames: int
            num_channels: int
            bits_per_sample: int
            encoding: str

        torchaudio.AudioMetaData = AudioMetaData  # type: ignore[attr-defined]

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]  # type: ignore[attr-defined]

    if not hasattr(torchaudio, "info"):
        def _info(path, backend=None):  # type: ignore[unused-argument]
            import soundfile as sf

            with sf.SoundFile(path) as f:
                sample_rate = int(f.samplerate)
                num_frames = int(len(f))
                num_channels = int(f.channels)
                subtype = (f.subtype or "").upper()
                bits = 0
                if subtype.startswith("PCM_"):
                    try:
                        bits = int(subtype.split("_", 1)[1])
                    except Exception:
                        bits = 0
                encoding = subtype or "UNKNOWN"
            return torchaudio.AudioMetaData(  # type: ignore[call-arg]
                sample_rate=sample_rate,
                num_frames=num_frames,
                num_channels=num_channels,
                bits_per_sample=bits,
                encoding=encoding,
            )

        torchaudio.info = _info  # type: ignore[attr-defined]

    warnings.filterwarnings(
        "ignore",
        message=r"(?s).*torchcodec is not installed correctly.*",
        category=UserWarning,
    )
    from pyannote.audio import Pipeline

    token = os.getenv("HUGGINGFACE_TOKEN")
    model_id = PINNED_DIARIZATION_MODEL_ID
    print(f"Loading diarization model: {model_id}")
    try:
        pipeline = Pipeline.from_pretrained(model_id, token=token)
    except TypeError:
        pipeline = Pipeline.from_pretrained(model_id, use_auth_token=token)
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
        print(f"⚠️ torchaudio.load failed, trying soundfile preload: {e}")

    try:
        import soundfile as sf
        import torch

        data, sample_rate = sf.read(audio_path, dtype="float32")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        else:
            data = data.T
        waveform = torch.from_numpy(data)
        return pipeline({"waveform": waveform, "sample_rate": int(sample_rate)}, **kwargs)
    except Exception as e:
        print(f"⚠️ Preloaded diarization input failed, falling back to file path: {e}")
        return pipeline(audio_path, **kwargs)
