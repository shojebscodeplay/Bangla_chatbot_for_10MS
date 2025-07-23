from langchain_core.prompts import PromptTemplate

CUSTOM_PROMPT_TEMPLATE = """<s>[INST]
You are a friendly, human-like AI assistant. Use **only the pdf db to answer the question**.
If the question is in Bangla, reply in Bangla.
If the question is in English, reply in English.
Answer in one short, clear, natural-sounding sentence.
Chat History:
{chat_history}

Context:
{context}

Question:
{question}
[/INST]"""

def get_prompt_template():
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question","chat_history"])
