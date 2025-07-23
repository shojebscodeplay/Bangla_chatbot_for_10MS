from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_get_history():
    session_id = "test_session1"
    
    # Create session with one query (note: queries as string)
    response = client.post("/api/ask", json={"queries": "What is ML?", "session_id": session_id})
    assert response.status_code == 200

    # Test history with JSON body
    response = client.post("/api/history", json={"session_id": session_id})
    assert response.status_code == 200
    data = response.json()
    
    assert data["session_id"] == session_id
    assert len(data["history"]) > 0
    assert data["history"][0]["query"] == "What is AI?"
