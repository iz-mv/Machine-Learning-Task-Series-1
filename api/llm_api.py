from pathlib import Path
from typing import List, Literal, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from llama_cpp import Llama

# загружаем модель
project_root = Path(__file__).resolve().parents[1]
model_path = project_root / "data" / "model.gguf"
if not model_path.exists():
    raise RuntimeError(f"Нет модели: {model_path}. Положи GGUF сюда.")

# настройки под мой ноут air m2 8gb
llm = Llama(
    model_path=str(model_path),
    n_ctx=2048, # можно 1536, если будет тесно по RAM
    n_gpu_layers=-1, # Metal ускорение
    n_batch=192, # 128–256
    seed=42
)

SYSTEM_DEFAULT = (
    "You are a helpful, concise assistant. "
    "Answer briefly unless the user asks for details. "
    "Use bullet points when listing."
)

# СХЕМЫ ВХОДА/ВЫХОДА
Role = Literal["system", "user", "assistant"]

class Message(BaseModel):
    role: Role
    content: str = Field(min_length=1)

class ChatRequest(BaseModel):
    messages: List[Message]  # диалог в формате как у chatgpt
    temperature: float = 0.3
    max_tokens: int = 512

class ChatResponse(BaseModel):
    reply: str

# FASTAPI
app = FastAPI(title="Local LLM API (llama.cpp)", version="1.0.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Мини-валидация: если нет system, добавим дефолтный
    msgs = req.messages
    if not msgs or msgs[0].role != "system":
        msgs = [Message(role="system", content=SYSTEM_DEFAULT)] + msgs

    try:
        out = llm.create_chat_completion(
            messages=[m.model_dump() for m in msgs],
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    reply = out["choices"][0]["message"]["content"].strip()
    return ChatResponse(reply=reply)
