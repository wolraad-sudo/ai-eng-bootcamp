import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI, OpenAIError
from pydantic import BaseModel

load_dotenv()

MODEL = "gpt-4o-mini"

app = FastAPI()


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    model: str


@app.get("/")
def hello():
    return {"message": "Hello, world!"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set in .env")

    client = OpenAI(api_key=api_key)
    try:
        response = client.responses.create(model=MODEL, input=req.question)
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"OpenAI request failed: {e}")

    return AskResponse(answer=response.output_text, model=MODEL)
