from transformers import BartForConditionalGeneration, BartTokenizer
import re

MODEL_NAME = "facebook/bart-large-cnn"

# Load model and tokenizer once at startup
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

def summarize_text(transcript_text):
    """
    Offline summarization using facebook/bart-large-cnn.
    Returns structured summary for web display.
    """

    if not transcript_text.strip():
        return "No text to summarize."

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
