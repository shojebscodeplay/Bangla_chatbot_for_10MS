from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from utils.logger import logger
from langchain_groq import ChatGroq
from utils.config import GROQ_API_KEY, DB_FAISS_PATH, EMBEDDING_MODEL, MODEL_NAME

def load_llm():
    if not GROQ_API_KEY:
        logger.error("❌ GROQ_API_KEY is missing. Set it in environment or .env file.")
        raise EnvironmentError("GROQ_API_KEY is not set.")
    return ChatGroq(
        model_name=MODEL_NAME,
        api_key=GROQ_API_KEY,
        temperature=0.5,
        max_tokens=600
    )
