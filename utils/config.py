import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_FAISS_PATH = r"C:\Users\Dell\Desktop\bots\new_bot\bangla_pdf_vector_store"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
MODEL_NAME = "llama3-70b-8192"  # Or "mistral-7b" if you want
# HF_TOKEN = os.getenv("HF_TOKEN")
# HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"