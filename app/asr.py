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
    task_mode = os.getenv("ASR_TASK", "translate").strip() or "translate"
    initial_prompt = os.getenv(
        "ASR_INITIAL_PROMPT",
        "This is a meeting recording with Indian speakers. Preserve names, numbers and technical terms accurately."
    ).strip() or None

    def _run_transcribe(relaxed=False):
        if not relaxed:
            decode = dict(
                beam_size=6,
                best_of=6,
                temperature=0.0,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 450},
                condition_on_previous_text=True,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
            )
        else:
            decode = dict(
                beam_size=5,
                best_of=5,
                temperature=[0.0, 0.2, 0.4],
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 250},
                condition_on_previous_text=False,
                no_speech_threshold=0.35,
                compression_ratio_threshold=2.8,
                log_prob_threshold=-1.5,
            )

        segments, info = model.transcribe(
            audio_path,
            task=task_mode,
            language=None,
            word_timestamps=True,
            initial_prompt=initial_prompt,
            **decode,
        )

        result = []
        speech_seconds = 0.0
        for seg in segments:
            words = []
            if getattr(seg, "words", None):
                for w in seg.words:
                    word_text = (getattr(w, "word", "") or "").strip()
                    if not word_text:
                        continue
                    words.append(
                        {
                            "word": word_text,
                            "start": getattr(w, "start", None),
                            "end": getattr(w, "end", None),
                            "probability": getattr(w, "probability", None),
                        }
                    )

            seg_start = float(seg.start)
            seg_end = float(seg.end)
            if seg_end > seg_start:
                speech_seconds += seg_end - seg_start

            result.append(
                {
                    "start": seg_start,
                    "end": seg_end,
                    "text": seg.text,
                    "words": words,
                }
            )

        total_duration = float(getattr(info, "duration", 0.0) or 0.0)
        coverage = (speech_seconds / total_duration) if total_duration > 0 else 1.0
        text_chars = sum(len((x.get("text") or "").strip()) for x in result)
        return result, coverage, total_duration, text_chars

    first_result, first_coverage, total_dur, first_chars = _run_transcribe(relaxed=False)
    print(
        f"ASR pass1: segments={len(first_result)}, chars={first_chars}, "
        f"coverage={first_coverage:.2f}, duration={total_dur:.2f}s"
    )

    should_retry = (
        total_dur >= 45.0
        and (first_coverage < 0.42 or first_chars < max(120, int(total_dur * 2.5)))
    )
    if not should_retry:
        return first_result

    second_result, second_coverage, _, second_chars = _run_transcribe(relaxed=True)
    print(
        f"ASR pass2: segments={len(second_result)}, chars={second_chars}, "
        f"coverage={second_coverage:.2f}"
    )

    first_score = (first_coverage * 0.6) + (min(first_chars, 4000) / 4000.0 * 0.4)
    second_score = (second_coverage * 0.6) + (min(second_chars, 4000) / 4000.0 * 0.4)
    return second_result if second_score > first_score else first_result


def transcribe_live_chunk(model, audio_path):
    """
    Low-latency transcription for short live-recording chunks.
    Returns plain text for immediate UI rendering.
    """
    task_mode = os.getenv("ASR_TASK", "translate").strip() or "translate"

    segments, _ = model.transcribe(
        audio_path,
        task=task_mode,
        language=None,
        beam_size=2,
        best_of=2,
        temperature=0.0,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 180},
        condition_on_previous_text=False,
        no_speech_threshold=0.4,
        compression_ratio_threshold=2.8,
        log_prob_threshold=-1.5,
        word_timestamps=False,
    )

    lines = []
    for seg in segments:
        text = (getattr(seg, "text", "") or "").strip()
        if text:
            lines.append(text)

    return " ".join(lines).strip()
