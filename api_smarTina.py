#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from smarTina_app_vector_ticket_db_api import smarTina_chat

load_dotenv()
API_HOST   = os.getenv("API_HOST", "0.0.0.0")
API_PORT   = int(os.getenv("API_PORT", "8080"))
API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"

app = FastAPI(title="API SmarTina")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    user_id: str = "default"

class ChatResponse(BaseModel):
    reply: str

@app.post("/chat/", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        return {"reply": "Per favore invia un messaggio non vuoto."}
    reply = smarTina_chat(req.user_id, req.message)
    return {"reply": reply}

@app.delete("/chat/{user_id}")
async def reset_session(user_id: str):
    from smarTina_app_vector_ticket_db_api import chiudi_sessione
    chiudi_sessione(user_id)
    return {"status": "sessione resettata"}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_smarTina:app", host=API_HOST, port=API_PORT, reload=API_RELOAD)
