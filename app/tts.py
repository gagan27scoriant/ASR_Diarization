from __future__ import annotations

import os
import uuid
from typing import Any

from gtts import gTTS

from app.config import AUDIO_FOLDER


TTS_LANGUAGE_ALIASES = {
    "english": "en",
    "en": "en",
    "eng_latn": "en",
    "hindi": "hi",
    "hi": "hi",
    "hin_deva": "hi",
    "tamil": "ta",
    "ta": "ta",
    "tam_taml": "ta",
    "telugu": "te",
    "te": "te",
    "tel_telu": "te",
    "kannada": "kn",
    "kn": "kn",
    "kan_knda": "kn",
    "malayalam": "ml",
    "ml": "ml",
    "mal_mlym": "ml",
    "marathi": "mr",
    "mr": "mr",
    "mar_deva": "mr",
    "gujarati": "gu",
    "gu": "gu",
    "guj_gujr": "gu",
    "bengali": "bn",
    "bn": "bn",
    "ben_beng": "bn",
    "punjabi": "pa",
    "pa": "pa",
    "pan_guru": "pa",
    "urdu": "ur",
    "ur": "ur",
    "urd_arab": "ur",
    "arabic": "ar",
    "ar": "ar",
    "ara_arab": "ar",
    "french": "fr",
    "fr": "fr",
    "fra_latn": "fr",
    "german": "de",
    "de": "de",
    "deu_latn": "de",
    "spanish": "es",
    "es": "es",
    "spa_latn": "es",
    "portuguese": "pt",
    "pt": "pt",
    "por_latn": "pt",
    "russian": "ru",
    "ru": "ru",
    "rus_cyrl": "ru",
}


def resolve_tts_lang(lang_value: str | None) -> str:
    raw = (lang_value or "").strip()
    if not raw:
        return "en"
    normalized = raw.lower().replace("-", "_").replace(" ", "")
    return TTS_LANGUAGE_ALIASES.get(normalized, raw.lower())


def synthesize_speech(text: str, lang_value: str | None = None, slow: bool = False) -> dict[str, Any]:
    content = str(text or "").strip()
    if not content:
        raise ValueError("Text is required for text-to-speech")

    lang = resolve_tts_lang(lang_value)
    filename = f"tts_{uuid.uuid4().hex[:12]}.mp3"
    output_path = os.path.join(AUDIO_FOLDER, filename)

    try:
        tts = gTTS(text=content, lang=lang, slow=bool(slow))
        tts.save(output_path)
    except ValueError as e:
        raise ValueError(f"Unsupported gTTS language '{lang}'") from e
    except Exception as e:
        raise RuntimeError(
            "gTTS synthesis failed. gTTS usually requires internet access at runtime."
        ) from e

    return {
        "audio_file": filename,
        "audio_url": f"/audio/{filename}",
        "tts_lang": lang,
        "text": content,
    }
