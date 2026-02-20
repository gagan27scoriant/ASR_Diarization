import re
import os

import requests
from transformers import BartForConditionalGeneration, BartTokenizer

MODEL_NAME = "facebook/bart-large-cnn"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
SUMMARY_BACKEND = os.getenv("SUMMARY_BACKEND", "ollama").lower()
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "mistral")
SUMMARY_TIMEOUT_SECONDS = int(os.getenv("SUMMARY_TIMEOUT_SECONDS", "10800"))

# Load BART only if selected
tokenizer = None
model = None
if SUMMARY_BACKEND == "bart":
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)


def _build_prompt(
    transcript_text: str,
    meeting_title: str,
    meeting_date: str,
    meeting_place: str
) -> str:
    return f"""
Create Minutes of a Meeting in this exact structure and heading order:
MINUTES OF A MEETING
TITLE :
DATE :
PLACE :

INTRODUCTION
ATTENDEES
SUMMARY OF THE MEETING
KEY ASPECTS DISCUSSED :
ACTION ITEMS AND ASSIGNED TO:
DEADLINES FOR THE TASKS:
THANK YOU

Use only these user-provided values for header fields:
- TITLE : {meeting_title}
- DATE : [{meeting_date}]
- PLACE : [{meeting_place}]

Rules:
- Do NOT include any transcript section.
- Do NOT include timestamps.
- Do NOT add explanations.
- If any section data is missing, output exactly "Not Applicable".
- Keep each section concise with bullet points where appropriate.

Conversation:
{transcript_text} """


def _summarize_with_ollama(
    transcript_text: str,
    meeting_title: str,
    meeting_date: str,
    meeting_place: str
) -> str:
    prompt = _build_prompt(transcript_text, meeting_title, meeting_date, meeting_place)
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": SUMMARY_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=SUMMARY_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    result = response.json()
    return result.get("response", "").strip()


def _summarize_with_bart(
    transcript_text: str,
    meeting_title: str,
    meeting_date: str,
    meeting_place: str
) -> str:
    # Step 1: Preprocess transcript
    transcript_text = transcript_text.replace("\n", " ").strip()

    # Step 2: Chunk transcript for BART
    max_chunk_length = 1000
    words = transcript_text.split()
    chunks = [" ".join(words[i:i+max_chunk_length]) for i in range(0, len(words), max_chunk_length)]

    # Step 3: Summarize each chunk
    chunk_summaries = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=5,
            max_length=150,
            early_stopping=True
        )
        chunk_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        chunk_summaries.append(chunk_summary)

    combined_summary = " ".join(chunk_summaries)

    # Step 4: Extract attendees (simple heuristic)
    attendees = sorted(set(re.findall(r"(?:Speaker\d+|[A-Z][a-z]+(?: [A-Z][a-z]+)*)", transcript_text)))

    # Step 5: Extract main topics (top 3 sentences)
    sentences = re.split(r'(?<=[.!?]) +', combined_summary)
    main_topics = sentences[:3]

    # Step 6: Extract actions (sentences with 'will' or 'should')
    actions = [s for s in sentences if re.search(r'\b(will|should)\b', s, re.IGNORECASE)]

    attendee_lines = "\n".join(f"- {name}" for name in attendees[:6]) if attendees else "Not Applicable"
    topic_lines = "\n".join(f"- {topic}" for topic in main_topics if topic.strip()) if main_topics else "Not Applicable"
    action_lines = "\n".join(f"- {action}" for action in actions if action.strip()) if actions else "Not Applicable"

    # Step 7: Assemble structured summary (for web display)
    structured_summary = f"""
MINUTES OF A MEETING
TITLE : {meeting_title}
DATE : [{meeting_date}]
PLACE : [{meeting_place}]

INTRODUCTION
Not Applicable

ATTENDEES
{attendee_lines}

SUMMARY OF THE MEETING
Not Applicable

KEY ASPECTS DISCUSSED :
{topic_lines}

ACTION ITEMS AND ASSIGNED TO:
{action_lines}

DEADLINES FOR THE TASKS:
Not Applicable

THANK YOU
"""

    return structured_summary.strip()

def _strip_transcript_section(summary_text: str) -> str:
    # Safety: remove transcript section if model adds it despite prompt constraints.
    cleaned = re.sub(
        r"(?is)\nTRANSCRIPTS OF THE MEETING:.*$",
        "",
        summary_text or "",
    )
    return cleaned.strip()


def summarize_text(transcript_text, meeting_title, meeting_date, meeting_place):
    """
    Summarize transcript text.
    Backends:
    - ollama (default): set SUMMARY_MODEL=mistral or SUMMARY_MODEL=llama3
    - bart: fallback local summarizer
    """

    if not transcript_text.strip():
        return "No text to summarize."

    try:
        if SUMMARY_BACKEND == "ollama":
            raw = _summarize_with_ollama(
                transcript_text, meeting_title, meeting_date, meeting_place
            )
            return _strip_transcript_section(raw)
        raw = _summarize_with_bart(
            transcript_text, meeting_title, meeting_date, meeting_place
        )
        return _strip_transcript_section(raw)
    except Exception as e:
        print("Summarization error:", e)
        return "Summary generation failed."
