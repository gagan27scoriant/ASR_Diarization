import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma:2b"

def summarize_text(transcript_text):
    if not transcript_text.strip():
        return "No text to summarize."

    prompt = f"""
You are an expert meeting analyst.

Create a concise professional summary in paragraphs.

Make a Bullet point summary of the meeting with the following structure:
The First paragraph should be a one-line summary of the meeting.
the Second paragraph should The Attendees who have attended the meeting and who was the host.
The Third paragraph should summarize the main topics, key decisions.
The fourth paragraph should summarize important discussions, and outcomes of the meeting.
The Fifth paragraph should be a what are the action items and who is responsible for each action item.
The sixth paragraph should be who was the best speaker in the meeting and why.


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
