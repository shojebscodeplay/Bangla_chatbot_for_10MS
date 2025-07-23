# utils/storage.py
import json
import os

def save_asked_questions(responses, session_id):
    os.makedirs("sessions", exist_ok=True)
    file_path = f"sessions/{session_id}.json"

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            existing = json.load(f)
    else:
        existing = []

    existing.extend(responses)

    with open(file_path, "w") as f:
        json.dump(existing, f, indent=2)
