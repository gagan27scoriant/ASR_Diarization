import os


AUDIO_FOLDER = "audio"
VIDEO_FOLDER = "videos"
OUTPUT_FOLDER = "outputs"
DOCUMENT_FOLDER = "documents"
RECORDINGS_FOLDER = "recordings"

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".aac", ".aiff", ".wma", ".amr", ".opus"
}
SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".mpeg", ".3gp"
}
SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".txt"}

MEDIA_FORMAT_ERROR = (
    "Unsupported media format. Audio: .wav, .mp3, .aac, .aiff, .wma, .amr, .opus | "
    "Video: .mp4, .mkv, .avi, .mov, .wmv, .mpeg, .3gp"
)
DOCUMENT_FORMAT_ERROR = "Unsupported document format. Use: .pdf, .docx, .txt"


def ensure_workspace_folders():
    for folder in (AUDIO_FOLDER, VIDEO_FOLDER, OUTPUT_FOLDER, DOCUMENT_FOLDER, RECORDINGS_FOLDER):
        os.makedirs(folder, exist_ok=True)


def is_supported_audio(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in SUPPORTED_AUDIO_EXTENSIONS


def is_supported_video(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in SUPPORTED_VIDEO_EXTENSIONS


def is_supported_media(filename: str) -> bool:
    return is_supported_audio(filename) or is_supported_video(filename)


def is_supported_document(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in SUPPORTED_DOCUMENT_EXTENSIONS
