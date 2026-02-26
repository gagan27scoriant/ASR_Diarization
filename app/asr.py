import os

import torch
from faster_whisper import WhisperModel


def load_asr(model_size=None):
    model_path = os.getenv("ASR_MODEL_PATH", "").strip()
    model_size = model_size or os.getenv("ASR_MODEL_SIZE", "medium")
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


def _resolve_asr_task():
    task_mode = (os.getenv("ASR_TASK", "transcribe") or "").strip().lower()
    return task_mode if task_mode in {"transcribe", "translate"} else "transcribe"


def _is_multilingual_mode():
    return (os.getenv("ASR_MULTILINGUAL", "1") or "").strip().lower() in {"1", "true", "yes", "on"}


def _is_english_output_mode():
    # Default to English output as requested; set ASR_OUTPUT_ENGLISH=0 to disable.
    return (os.getenv("ASR_OUTPUT_ENGLISH", "1") or "").strip().lower() in {"1", "true", "yes", "on"}


def _detect_language_hint(model, audio_path):
    forced_lang = (os.getenv("ASR_LANGUAGE", "") or "").strip().lower()
    if forced_lang:
        return forced_lang

    auto_detect = (os.getenv("ASR_AUTO_LANGUAGE", "1") or "").strip().lower() in {"1", "true", "yes", "on"}
    if not auto_detect:
        return None

    min_prob = float(os.getenv("ASR_AUTO_LANGUAGE_MIN_PROB", "0.55") or 0.55)
    try:
        probe_segments, probe_info = model.transcribe(
            audio_path,
            task="transcribe",
            language=None,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            vad_filter=False,
            condition_on_previous_text=False,
            word_timestamps=False,
        )
        # Materialize a tiny amount so backend runs language probe reliably.
        try:
            next(iter(probe_segments))
        except StopIteration:
            pass

        lang = (getattr(probe_info, "language", None) or "").strip().lower()
        prob = float(getattr(probe_info, "language_probability", 0.0) or 0.0)
        if lang and prob >= min_prob:
            print(f"ASR language locked: {lang} (p={prob:.2f})")
            return lang
        print(f"ASR language auto-detect weak: lang={lang or 'unknown'}, p={prob:.2f}")
    except Exception as e:
        print(f"⚠️ ASR language detection failed: {e}")
    return None


def transcribe(model, audio_path):
    """
    Transcribe audio and return Whisper-like output.
    """
    multilingual_mode = _is_multilingual_mode()
    task_mode = "translate" if _is_english_output_mode() else _resolve_asr_task()

    language_hint = None if multilingual_mode else _detect_language_hint(model, audio_path)
    initial_prompt = (
        (os.getenv("ASR_INITIAL_PROMPT", "") or "").strip()
        or "Translate to clear, natural English while preserving exact meaning, intent, names, and numbers."
    )

    def _run_transcribe(mode="strict"):
        if mode == "strict":
            decode = dict(
                beam_size=5,
                best_of=5,
                temperature=0.0,
                vad_filter=False,
                condition_on_previous_text=False if multilingual_mode else True,
                no_speech_threshold=0.45,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.2,
            )
        elif mode == "relaxed":
            decode = dict(
                beam_size=4,
                best_of=4,
                temperature=[0.0, 0.2, 0.4],
                vad_filter=False,
                condition_on_previous_text=False,
                no_speech_threshold=0.25,
                compression_ratio_threshold=2.8,
                log_prob_threshold=-1.8,
            )
        else:
            # Recovery pass for long audio where VAD may trim late segments.
            decode = dict(
                beam_size=3,
                best_of=3,
                temperature=[0.0, 0.2, 0.4],
                vad_filter=False,
                condition_on_previous_text=False,
                no_speech_threshold=1.0,
                compression_ratio_threshold=3.2,
                log_prob_threshold=-2.2,
            )

        segments, info = model.transcribe(
            audio_path,
            task=task_mode,
            language=language_hint,
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
                    "text": " ".join((seg.text or "").split()),
                    "words": words,
                }
            )

        total_duration = float(getattr(info, "duration", 0.0) or 0.0)
        coverage = (speech_seconds / total_duration) if total_duration > 0 else 1.0
        text_chars = sum(len((x.get("text") or "").strip()) for x in result)
        last_end = float(result[-1]["end"]) if result else 0.0
        span_ratio = (last_end / total_duration) if total_duration > 0 else 1.0
        return result, coverage, total_duration, text_chars, span_ratio

    def _score(coverage, text_chars, total_dur, span_ratio):
        density = (text_chars / max(1.0, total_dur * 5.0)) if total_dur > 0 else 1.0
        density = max(0.0, min(1.0, density))
        return (coverage * 0.45) + (span_ratio * 0.35) + (density * 0.20)

    first_result, first_coverage, total_dur, first_chars, first_span_ratio = _run_transcribe(mode="strict")
    print(
        f"ASR pass1: segments={len(first_result)}, chars={first_chars}, "
        f"coverage={first_coverage:.2f}, span={first_span_ratio:.2f}, duration={total_dur:.2f}s, "
        f"mode={'english-translate' if task_mode == 'translate' else 'native-transcribe'}"
    )

    need_relaxed_retry = (
        total_dur >= 45.0
        and (
            first_coverage < 0.42
            or first_span_ratio < 0.75
            or first_chars < max(120, int(total_dur * 2.5))
        )
    )
    candidates = [
        {
            "name": "pass1",
            "result": first_result,
            "coverage": first_coverage,
            "chars": first_chars,
            "span_ratio": first_span_ratio,
            "score": _score(first_coverage, first_chars, total_dur, first_span_ratio),
        }
    ]

    if need_relaxed_retry:
        second_result, second_coverage, _, second_chars, second_span_ratio = _run_transcribe(mode="relaxed")
        print(
            f"ASR pass2: segments={len(second_result)}, chars={second_chars}, "
            f"coverage={second_coverage:.2f}, span={second_span_ratio:.2f}"
        )
        candidates.append(
            {
                "name": "pass2",
                "result": second_result,
                "coverage": second_coverage,
                "chars": second_chars,
                "span_ratio": second_span_ratio,
                "score": _score(second_coverage, second_chars, total_dur, second_span_ratio),
            }
        )

    best = max(candidates, key=lambda x: x["score"])

    need_full_retry = (
        total_dur >= 45.0
        and (best["coverage"] < 0.55 or best["span_ratio"] < 0.88)
    )
    if need_full_retry:
        third_result, third_coverage, _, third_chars, third_span_ratio = _run_transcribe(mode="full")
        print(
            f"ASR pass3 (full): segments={len(third_result)}, chars={third_chars}, "
            f"coverage={third_coverage:.2f}, span={third_span_ratio:.2f}"
        )
        candidates.append(
            {
                "name": "pass3",
                "result": third_result,
                "coverage": third_coverage,
                "chars": third_chars,
                "span_ratio": third_span_ratio,
                "score": _score(third_coverage, third_chars, total_dur, third_span_ratio),
            }
        )
        best = max(candidates, key=lambda x: x["score"])

    print(
        f"ASR selected {best['name']}: coverage={best['coverage']:.2f}, "
        f"span={best['span_ratio']:.2f}, chars={best['chars']}"
    )
    return best["result"]


def transcribe_live_chunk(model, audio_path):
    """
    Low-latency transcription for short live-recording chunks.
    Returns plain text for immediate UI rendering.
    """
    multilingual_mode = _is_multilingual_mode()
    task_mode = "translate" if _is_english_output_mode() else _resolve_asr_task()
    language_hint = None if multilingual_mode else ((os.getenv("ASR_LANGUAGE", "") or "").strip().lower() or None)

    segments, _ = model.transcribe(
        audio_path,
        task=task_mode,
        language=language_hint,
        beam_size=2,
        best_of=2,
        temperature=0.0,
        vad_filter=False,
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
