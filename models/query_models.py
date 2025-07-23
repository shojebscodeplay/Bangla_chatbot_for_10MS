from pydantic import BaseModel

class QueryRequest(BaseModel):
    session_id: str
    queries: str

class SessionRequest(BaseModel):
    session_id: str