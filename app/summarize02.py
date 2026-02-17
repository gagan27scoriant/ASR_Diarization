import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma2:2b"

def summarize_text(transcript_text):
    if not transcript_text.strip():
        return "No text to summarize."

    prompt = f"""
take the input from both the conversation identifying the speakers and make it a small summary where an normal human also shud understand seeing the summary.

OUTPUT FORMAT (FOLLOW EXACTLY):

One Paragraph Meaningful Summary:
<one Paragraph meaningful summary>

# Attendees:
# - List all speakers or participants in the conversation

# identify the Main Topics and Key Decisions:
# - Bullet points

# identify the Plan of Actions and the Responsible Person for tht tasks:
# - Action → Responsible person

# Best Speaker:
# - Name of speaker
# - Reason why

if any details not found leave it blanck nut not spoil the summary

Conversation:
{transcript_text}
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()
        result = response.json()

        return result.get("response", "").strip()

    except Exception as e:
        print("Llama3 Error:", e)
        return "Summary generation failed."
