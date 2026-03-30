import os
import warnings

import torch
import torchaudio

PINNED_DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-community-1"

class DiarizationHandle:
    def __init__(self, backend: str, model, meta: dict | None = None):
        self.backend = backend
        self.model = model
        self.meta = meta or {}


class _SimpleTurn:
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end


class _SimpleDiarization:
    def __init__(self, segments):
        self._segments = segments

    def itertracks(self, yield_label=True):
        for seg in self._segments:
            turn = _SimpleTurn(seg["start"], seg["end"])
            if yield_label:
                yield turn, None, seg["speaker"]
            else:
                yield turn, None


class _DiarizationResult:
    def __init__(self, segments):
        self.speaker_diarization = _SimpleDiarization(segments)


def _load_pyannote():
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

    return DiarizationHandle("pyannote", pipeline)


def _load_nemo():
    config_path = (os.getenv("NEMO_DIAR_CONFIG") or "").strip()
    if not config_path:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        default_cfg = os.path.join(repo_root, "configs", "nemo_diarization.yaml")
        if os.path.isfile(default_cfg):
            config_path = default_cfg
        else:
            raise ValueError("NEMO_DIAR_CONFIG is required for NeMo diarization backend.")

    from omegaconf import OmegaConf
    from nemo.collections.asr.models import ClusteringDiarizer

    cfg = OmegaConf.load(config_path)
    out_dir = (os.getenv("NEMO_DIAR_OUT_DIR") or "outputs/nemo_diarization").strip()
    os.makedirs(out_dir, exist_ok=True)
    cfg.diarizer.out_dir = out_dir

    diarizer = ClusteringDiarizer(cfg=cfg)
    return DiarizationHandle("nemo", diarizer, {"out_dir": out_dir, "config_path": config_path})


def load_diarization():
    backend = (os.getenv("DIARIZATION_BACKEND") or "").strip().lower()
    if not backend:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        default_cfg = os.path.join(repo_root, "configs", "nemo_diarization.yaml")
        backend = "nemo" if os.path.isfile(default_cfg) else "pyannote"
    if backend == "nemo":
        return _load_nemo()
    return _load_pyannote()



def _diarize_nemo(handle: DiarizationHandle, audio_path: str):
    diarizer = handle.model
    out_dir = handle.meta.get("out_dir") or "outputs/nemo_diarization"

    manifest_path = os.path.join(out_dir, "manifest.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(
            f'{{"audio_filepath": "{audio_path}", "offset": 0, "duration": null, "label": "infer", "text": ""}}\\n'
        )

    # Update manifest path dynamically per audio file.
    if hasattr(diarizer, "cfg"):
        try:
            diarizer.cfg.diarizer.manifest_filepath = manifest_path
        except Exception:
            pass
    if hasattr(diarizer, "_diarizer_params"):
        try:
            diarizer._diarizer_params.manifest_filepath = manifest_path
        except Exception:
            pass

    diarizer.diarize(paths2audio_files=[audio_path])

    pred_dir = os.path.join(out_dir, "pred_rttms")
    base = os.path.splitext(os.path.basename(audio_path))[0]
    rttm_candidates = []
    if os.path.isdir(pred_dir):
        for name in os.listdir(pred_dir):
            if not name.endswith(".rttm"):
                continue
            if base in name:
                rttm_candidates.append(os.path.join(pred_dir, name))
    if not rttm_candidates:
        # Fallback to any RTTM file if name-matching fails.
        rttm_candidates = [
            os.path.join(pred_dir, name)
            for name in os.listdir(pred_dir)
            if name.endswith(".rttm")
        ]
    if not rttm_candidates:
        raise RuntimeError("NeMo diarization did not produce any RTTM output.")

    rttm_path = sorted(rttm_candidates)[0]
    segments = []
    with open(rttm_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or not line.startswith("SPEAKER"):
                continue
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            start = float(parts[3])
            dur = float(parts[4])
            speaker = parts[7]
            segments.append(
                {"start": start, "end": start + dur, "speaker": speaker}
            )

    return _DiarizationResult(segments)


def diarize(pipeline, audio_path):
    if isinstance(pipeline, DiarizationHandle) and pipeline.backend == "nemo":
        return _diarize_nemo(pipeline, audio_path)

    pipeline = pipeline.model if isinstance(pipeline, DiarizationHandle) else pipeline
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
