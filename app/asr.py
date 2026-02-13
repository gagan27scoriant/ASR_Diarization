import whisper


def load_asr(model_size="base"):
    """
    Load Whisper model (CPU).
    Use 'tiny' for faster CPU inference.
    """
    return whisper.load_model(model_size)


def transcribe(model, audio_path):
    """
    Transcribe audio and return Whisper output.
    """
    return model.transcribe(audio_path)
