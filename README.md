# 📘 Bangla PDF Question Answering System (RAG-based QA) Assignment of 10 Minute School


A modular Retrieval-Augmented Generation (RAG) based system for answering questions from Bangla literature, specifically tested on Rabindranath Tagore's famous story **"অপরিচিতা"**.
This project integrates PDF parsing, Bangla OCR preprocessing, chunking, semantic search using FAISS, and LLM-based response generation with memory. 

---

## 🚀 Project Motivation
The original text was highly fragmented and contained many MCQs, making it difficult for the vector database to perform accurate similarity search.
Therefore, I used a clean PDF version of "অপরিচিতা" instead. Still there was problem in text so i use online OCR and get better texts.

Due to the limitations in Bangla language support in vector embedding and OCR, it is often difficult to build reliable question-answering systems. This project is an attempt to:

- Extract Bangla text from scanned/unstructured PDFs.
- Preprocess and clean noisy or broken OCR outputs.
- Chunk the cleaned data for better semantic matching.
- Answer user queries using a multilingual embedding model and a powerful LLM (`LLaMA3-70B`).

---

## 🛠️ Key Features

- ✅ Bangla OCR + preprocessing pipeline
- ✅ Trial-and-error-based optimal chunking strategy
- ✅ Vector similarity search using **FAISS**
- ✅ Embedding Model: `intfloat/multilingual-e5-base`
- ✅ LLM Model: `llama3-70b-8192` with memory buffer for contextual QA
- ✅ Modular and extendable structure
- ✅ Compatible with other Bangla texts by updating PDF

## 🛠️ Improvement Possibilities

By using an agentic RAG approach, I can integrate external tools for search, which will help reduce hallucinations and improve answer accuracy.

## 🙋‍♂️ Confession

I am not a professional coder, just a curious learner. I took help from various AI applications, but I believe that with time and good guidance, coding will become a habit and much easier for me.
r me.

