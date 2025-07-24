import os
import re
import logging
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import Tuple, List
import torch    
import unicodedata

# ✅ Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Suppress unnecessary warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ✅ Extract text from PDF (with error handling)
def extract_text(pdf_path: str) -> Tuple[str, int]:
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text("text") for page in doc])
        return text, len(doc)
    except Exception as e:
        logger.error(f"Failed to extract text: {e}")
        raise

# ✅ Improved Bangla text cleaning
def clean_text(text: str) -> str:
    # Unicode Normalization (solve: ো -> ে +া etc.)
    text = unicodedata.normalize("NFC", text)

    # Remove control characters
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)

    # Remove non-Bangla letters, keep punctuation
    text = re.sub(r'[^\u0980-\u09FF\s\.,!?()\-—"\'৳ঃ;ঃঃ]', '', text)

    # Remove isolated Bangla digits
    text = re.sub(r'(\s)[০-৯]+(\s)', r'\1\2', text)

    # Remove multiple newlines and extra spaces
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    text = re.sub(r"\s+", " ", text)

    # Fix broken consonant conjuncts (e.g., ক ্ ষ => ক্ষ)
    text = re.sub(r"([অ-হা-ৌড়ঢ়য়])\s+([্])\s+([ক-হ])", r"\1\2\3", text)

    # Fix broken vowel signs (e.g., ম ি => মি)
    text = re.sub(r"([অ-হ])\s+([া-ৌ])", r"\1\2", text)

    # Remove repeated diacritic signs (েে -> ে)
    text = re.sub(r'(ে){2,}', 'ে', text)
    text = re.sub(r'(া){2,}', 'া', text)
    text = re.sub(r'(ি){2,}', 'ি', text)
    text = re.sub(r'(ু){2,}', 'ু', text)
    text = re.sub(r'(়){2,}', '়', text)

    # Remove leading/trailing punctuation & extra dots
    text = re.sub(r"[।]+", "।", text)
    text = re.sub(r"([।!?]){2,}", r"\1", text)
    text = re.sub(r'^[\.\s]+|[\.\s]+$', '', text)

    # Final trim
    return text.strip()

# ✅ Check if chunk is Bangla-dominant (improved logic)
def is_chunk_clean(text: str, threshold: float = 0.8) -> bool:
    bangla_chars = re.findall(r'[\u0980-\u09FF]', text)
    return (len(bangla_chars) / max(len(text), 1)) >= threshold

# ✅ Optimized chunking with overlap and semantic boundaries
def split_documents(text: str) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,  # Larger chunks for Bangla (fewer tokens per word)
        chunk_overlap=200,
        separators=["\n\n", "\n", "।", "?", "!", "…", ".",  " ", ""]  # Bangla sentence boundaries
    )
    return text_splitter.split_text(text)

# ✅ Main Execution
if __name__ == "__main__":
    pdf_path = r"C:\Users\Dell\Desktop\Multilingual_RAG_for_10MS\data\oporichita (1).pdf"
    
    # Step 1: Extract and clean text
    raw_text, total_pages = extract_text(pdf_path)
    logger.info(f"📄 PDF loaded: {total_pages} pages | {len(raw_text)} chars")
    
    cleaned_text = clean_text(raw_text)
    logger.info("🧹 Text cleaned successfully.")

    # Step 2: Split into semantic chunks
    documents = split_documents(cleaned_text)
    documents = [doc for doc in documents if is_chunk_clean(doc)]
    logger.info(f"✂️ Generated {len(documents)} clean Bangla chunks.")

    # Step 3: Use a better multilingual embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",  # Better for Bangla
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}  # Improves retrieval
    )

    # Step 4: Store in FAISS with compression
    db = FAISS.from_texts(documents, embedding_model)
    db.save_local("bangla_pdf_vector_store")
    logger.info("💾 FAISS vector store saved successfully!")