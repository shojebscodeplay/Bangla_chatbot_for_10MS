from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_ask_questions():
    response = client.post("/api/ask", json={
        "session_id": "test-session",
        "queries": "What is AI?"
    })
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert data["results"][0]["query"] == "What is AI?"



