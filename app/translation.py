import os
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEFAULT_NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"
DEFAULT_NLLB_LOCAL_DIR = os.path.join("models", "nllb-200-distilled-600M")

LANGUAGE_ALIASES = {
    "english": "eng_Latn",
    "en": "eng_Latn",
    "hindi": "hin_Deva",
    "hi": "hin_Deva",
    "tamil": "tam_Taml",
    "ta": "tam_Taml",
    "telugu": "tel_Telu",
    "te": "tel_Telu",
    "kannada": "kan_Knda",
    "kn": "kan_Knda",
    "malayalam": "mal_Mlym",
    "ml": "mal_Mlym",
    "marathi": "mar_Deva",
    "mr": "mar_Deva",
    "gujarati": "guj_Gujr",
    "gu": "guj_Gujr",
    "bengali": "ben_Beng",
    "bn": "ben_Beng",
    "punjabi": "pan_Guru",
    "pa": "pan_Guru",
    "urdu": "urd_Arab",
    "ar": "ara_Arab",
    "arabic": "ara_Arab",
    "french": "fra_Latn",
    "fr": "fra_Latn",
    "german": "deu_Latn",
    "de": "deu_Latn",
    "spanish": "spa_Latn",
    "es": "spa_Latn",
    "portuguese": "por_Latn",
    "pt": "por_Latn",
    "russian": "rus_Cyrl",
    "ru": "rus_Cyrl",
    "chinese": "zho_Hans",
    "zh": "zho_Hans",
    "japanese": "jpn_Jpan",
    "ja": "jpn_Jpan",
    "korean": "kor_Hang",
    "ko": "kor_Hang",
    "italian": "ita_Latn",
    "it": "ita_Latn",
}

_TRANSLATOR = None


def _is_enabled(flag_name: str, default: str = "1") -> bool:
    return (os.getenv(flag_name, default) or "").strip().lower() in {"1", "true", "yes", "on"}


def _expand_local_path(path_value: str) -> str:
    return os.path.abspath(os.path.expanduser(path_value))


def _get_preferred_local_model_path() -> str:
    configured = (os.getenv("NLLB_MODEL_PATH", "") or "").strip()
    if configured:
        return _expand_local_path(configured)
    legacy_local = _expand_local_path("nllb_model")
    if Path(legacy_local).is_dir():
        return legacy_local
    return _expand_local_path(DEFAULT_NLLB_LOCAL_DIR)


def _has_local_model(path_value: str) -> bool:
    path = Path(path_value)
    if not path.is_dir():
        return False
    has_weights = (
        (path / "model.safetensors").is_file()
        or (path / "pytorch_model.bin").is_file()
        or (path / "model.safetensors.index.json").is_file()
        or (path / "pytorch_model.bin.index.json").is_file()
        or any(path.glob("model-*.safetensors"))
        or any(path.glob("pytorch_model-*.bin"))
    )
    return (path / "config.json").is_file() and (path / "tokenizer_config.json").is_file() and has_weights


def _find_hf_cached_model_path(model_name: str) -> str:
    cache_home = (os.getenv("HF_HOME") or "").strip()
    if not cache_home:
        cache_home = os.path.expanduser("~/.cache/huggingface")
    hub_root = Path(cache_home) / "hub"
    repo_dir = hub_root / f"models--{model_name.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return ""

    candidates = sorted(
        [p for p in snapshots_dir.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if _has_local_model(str(candidate)):
            return str(candidate.resolve())
    return ""


def _has_meta_tensors(model) -> bool:
    for p in model.parameters():
        if getattr(p, "is_meta", False):
            return True
    for b in model.buffers():
        if getattr(b, "is_meta", False):
            return True
    return False


def download_nllb_model(model_name: str | None = None, target_dir: str | None = None) -> str:
    resolved_model_name = (model_name or os.getenv("NLLB_MODEL_NAME") or DEFAULT_NLLB_MODEL_NAME).strip()
    resolved_target = _expand_local_path(target_dir or _get_preferred_local_model_path())
    os.makedirs(resolved_target, exist_ok=True)

    print(f"Downloading NLLB model '{resolved_model_name}' into '{resolved_target}'...")
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_name, local_files_only=False, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        resolved_model_name,
        local_files_only=False,
        low_cpu_mem_usage=False,
    )
    tokenizer.save_pretrained(resolved_target)
    model.save_pretrained(resolved_target)
    os.environ["NLLB_MODEL_PATH"] = resolved_target
    print(f"NLLB model downloaded. NLLB_MODEL_PATH='{resolved_target}'")
    return resolved_target


class NLLBTranslator:
    def __init__(self):
        self.model_name = (os.getenv("NLLB_MODEL_NAME", DEFAULT_NLLB_MODEL_NAME) or "").strip() or DEFAULT_NLLB_MODEL_NAME
        self.local_model_path = _get_preferred_local_model_path()
        self.offline_only = _is_enabled("NLLB_OFFLINE_ONLY", "1")
        self.auto_download = _is_enabled("NLLB_AUTO_DOWNLOAD", "0")
        self.default_source_lang = (os.getenv("NLLB_SOURCE_LANG", "eng_Latn") or "").strip() or "eng_Latn"
        self.max_input_tokens = int((os.getenv("NLLB_MAX_INPUT_TOKENS", "512") or "512").strip())
        self.max_new_tokens = int((os.getenv("NLLB_MAX_NEW_TOKENS", "512") or "512").strip())

        cached_model_path = _find_hf_cached_model_path(self.model_name)
        if _has_local_model(self.local_model_path):
            self.model_source = self.local_model_path
            self.offline_only = True
            os.environ["NLLB_MODEL_PATH"] = self.local_model_path
        elif cached_model_path:
            self.model_source = cached_model_path
            self.offline_only = True
            os.environ["NLLB_MODEL_PATH"] = cached_model_path
        elif self.auto_download:
            self.local_model_path = download_nllb_model(self.model_name, self.local_model_path)
            self.model_source = self.local_model_path
            self.offline_only = True
            os.environ["NLLB_MODEL_PATH"] = self.local_model_path
        else:
            self.model_source = self.model_name

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading NLLB translation model '{self.model_source}' on {self.device.upper()}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_source,
                local_files_only=self.offline_only,
                use_fast=False,
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_source,
                local_files_only=self.offline_only,
                low_cpu_mem_usage=False,
            )
        except Exception as e:
            if self.offline_only:
                raise RuntimeError(
                    "NLLB model not found offline. Set NLLB_MODEL_PATH to a local model directory "
                    "or pre-download the model into Hugging Face cache."
                ) from e
            raise

        try:
            if not getattr(self.model, "hf_device_map", None):
                self.model.to(self.device)
        except NotImplementedError as e:
            if "Cannot copy out of meta tensor" not in str(e):
                raise
            if _has_meta_tensors(self.model):
                raise RuntimeError(
                    "NLLB model has meta tensors (incomplete/invalid weights). "
                    "Re-download the model and point NLLB_MODEL_PATH to a complete folder."
                ) from e
            raise
        self.model.eval()
        self.lang_code_to_id = {}
        for token in (getattr(self.tokenizer, "additional_special_tokens", None) or []):
            token_id = int(self.tokenizer.convert_tokens_to_ids(token))
            if token_id >= 0:
                self.lang_code_to_id[token] = token_id

        if self.default_source_lang not in self.lang_code_to_id:
            self.default_source_lang = "eng_Latn"
        print("NLLB translation model loaded.")

    def resolve_lang_code(self, lang_value: str, is_target: bool = False) -> str:
        raw = (lang_value or "").strip()
        if not raw and not is_target:
            raw = self.default_source_lang
        if not raw:
            raise ValueError("Target language is required")

        if raw in self.lang_code_to_id:
            return raw

        key = raw.lower().replace("-", "_").strip()
        key_compact = key.replace(" ", "")
        mapped = LANGUAGE_ALIASES.get(key) or LANGUAGE_ALIASES.get(key_compact)
        if mapped and mapped in self.lang_code_to_id:
            return mapped

        raise ValueError(
            f"Unsupported language '{lang_value}'. Use language name (e.g., Hindi) or NLLB code (e.g., hin_Deva)."
        )

    def _translate_batch(self, lines: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        if not lines:
            return []

        self.tokenizer.src_lang = src_lang
        encoded = self.tokenizer(
            lines,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.device)

        generated = self.model.generate(
            **encoded,
            forced_bos_token_id=self.lang_code_to_id[tgt_lang],
            max_new_tokens=self.max_new_tokens,
            num_beams=4,
            length_penalty=1.0,
        )
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [" ".join((text or "").split()) for text in decoded]

    def translate_lines(self, lines: Iterable[str], target_lang: str, source_lang: str | None = None) -> list[str]:
        src = self.resolve_lang_code(source_lang or "", is_target=False)
        tgt = self.resolve_lang_code(target_lang, is_target=True)

        line_list = [str(x or "") for x in lines]
        translated = [""] * len(line_list)
        non_empty_idx = [i for i, x in enumerate(line_list) if x.strip()]
        if not non_empty_idx:
            return translated

        batch_size = int((os.getenv("NLLB_BATCH_SIZE", "8") or "8").strip())
        for start in range(0, len(non_empty_idx), max(1, batch_size)):
            idx_batch = non_empty_idx[start:start + max(1, batch_size)]
            text_batch = [line_list[i] for i in idx_batch]
            out_batch = self._translate_batch(text_batch, src, tgt)
            for idx, out in zip(idx_batch, out_batch):
                translated[idx] = out
        return translated

    def translate_text(self, text: str, target_lang: str, source_lang: str | None = None) -> str:
        if not (text or "").strip():
            return ""
        lines = text.split("\n")
        translated_lines = self.translate_lines(lines, target_lang=target_lang, source_lang=source_lang)
        return "\n".join(translated_lines)


def get_translator() -> NLLBTranslator:
    global _TRANSLATOR
    if _TRANSLATOR is None:
        _TRANSLATOR = NLLBTranslator()
    return _TRANSLATOR
