from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from utils.logger import logger
from utils.config import DB_FAISS_PATH, EMBEDDING_MODEL
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    try:
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        logger.error(f"❌ Failed to load FAISS DB: {e}")
        raise
