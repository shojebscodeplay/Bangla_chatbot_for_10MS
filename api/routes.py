from fastapi import APIRouter, HTTPException
from models.query_models import QueryRequest, SessionRequest
from service.llm_service import load_llm
from service.vectorstore_service import load_vectorstore
from service.qa_service import create_qa_chain
from utils.config import EMBEDDING_MODEL, DB_FAISS_PATH, GROQ_API_KEY
from utils.storage import save_asked_questions
import json
import os

router = APIRouter()

llm = load_llm()
vectorstore = load_vectorstore()

@router.post("/ask")
def ask_questions(request: QueryRequest):
    try:
        qa_chain = create_qa_chain(llm, vectorstore, request.session_id)

        # Use call or invoke with "question" key to keep memory updated
        result = qa_chain.invoke({"question": request.queries})

        response = {"query": request.queries, "result": result['answer']}

        save_asked_questions([response], request.session_id)

        return {"session_id": request.session_id, "results": [response]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/history")
def get_history(request: SessionRequest):
    session_id = request.session_id
    file_path = f"sessions/{session_id}.json"

    if not os.path.exists(file_path):
        return {"session_id": session_id, "history": []}

    try:
        with open(file_path, "r") as f:
            history = json.load(f)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading session: {str(e)}")
