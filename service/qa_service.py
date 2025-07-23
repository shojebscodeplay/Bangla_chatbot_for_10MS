from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from domain.prompt import get_prompt_template
import os
import json

memory_store = {}

def create_qa_chain(llm, vectorstore, session_id="default"):
    if session_id not in memory_store:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        # Load chat history from saved file if exists
        file_path = f"sessions/{session_id}.json"
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    saved = json.load(f)
                for pair in saved:
                    memory.chat_memory.add_user_message(pair["query"])
                    memory.chat_memory.add_ai_message(pair["result"])
            except Exception as e:
                print(f"Warning: Could not load previous memory for session {session_id}: {e}")
        memory_store[session_id] = memory

    memory = memory_store[session_id]

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": get_prompt_template()},
        return_source_documents=False,
    )
