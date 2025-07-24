from fastapi import FastAPI
from api.routes import router

app = FastAPI(title="LLM QA Assistant")

app.include_router(router, prefix="/api")
