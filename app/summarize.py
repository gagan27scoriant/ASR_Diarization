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


def _build_prompt(transcript_text: str) -> str:
    return f"""
Create a professional summary of this meeting: 
1. MEETING TITLE and DATE and TIME 
2. AJENDA OF THE MEETING 
3. MAIN TOPICS DISCUSSED 
4. ACTION ITEMS AND ASSIGNEES 
5. DEADLINE FOR ACTION ITEMS.

IF Anthing is missing, just write "Not Applicable" for that section. Just write the summary in a clear and concise manner, without any additional explanations or formatting and keep it within 6 Points for each section. if Anything Not Applicable, just write "Not Applicable" for that section with no further explanation.
Take the assignee name from the speaker names after the finalizing name and just mention their name without any further explanation. If there are multiple assignees, just write their names by Bullet Points. Take the following transcript of the meeting and create the summary based on the above structure:

Conversation:
{transcript_text} """


def _summarize_with_ollama(transcript_text: str) -> str:
    prompt = _build_prompt(transcript_text)
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


def _summarize_with_bart(transcript_text: str) -> str:
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
    attendees = list(set(re.findall(r"(?:Speaker\d+|[A-Z][a-z]+(?: [A-Z][a-z]+)*)", transcript_text)))

    # Step 5: Extract main topics (top 3 sentences)
    sentences = re.split(r'(?<=[.!?]) +', combined_summary)
    main_topics = sentences[:3]

    # Step 6: Extract actions (sentences with 'will' or 'should')
    actions = [s for s in sentences if re.search(r'\b(will|should)\b', s, re.IGNORECASE)]

    # Step 7: Assemble structured summary (for web display)
    structured_summary = f"""
One Paragraph Meaningful Summary:
{combined_summary}

# Attendees:
# - {"\n# - ".join(attendees)}

# Main Topics and Key Decisions:
# - {"\n# - ".join(main_topics)}

# Plan of Actions and Responsible Person:
# - {"\n# - ".join(actions)}

# Best Speaker:
# - {attendees[0] if attendees else ""}
# - Reason: Most active in conversation
"""

    return structured_summary.strip()

def summarize_text(transcript_text):
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
            return _summarize_with_ollama(transcript_text)
        return _summarize_with_bart(transcript_text)
    except Exception as e:
        print("Summarization error:", e)
        return "Summary generation failed."
