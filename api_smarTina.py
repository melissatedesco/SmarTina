#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# api_smarTina.py
# ciao
# ciao2

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
# import json
# from fastapi import HTTPException
# from smarTina_app_vector_ticket_db_api import(
# smarTina_chat,
# carica_storia_temp,
# salva_messaggio_temp
# ) 
# ciao

# === CONFIG ===
load_dotenv()
API_HOST = os.getenv("API_HOST")
API_PORT = int(os.getenv("API_PORT"))
API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"

# === APP ===
app = FastAPI(title="API SmarTina")

# CORS (per Angular)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in prod: specificare dominio Angular
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === MODEL ===
class ChatRequest(BaseModel):
    message: str
    history: list = []  # [{"role": "user", "content": "..."}]

class ChatResponse(BaseModel):
    reply: str

# === ENDPOINT ===
@app.post("/chat/")
async def chat(user_message: ChatRequest):
    if not user_message.message.strip():
        return {"reply": "Per favore invia un messaggio non vuoto."}
    
    user_id = "default"
    # reply = smarTina_chat(user_id, user_message.message, user_message.history)
    reply = f"Hai detto: {user_message.message}"

    print("DEBUG: user_message =", user_message.message)
    print("DEBUG: reply =", reply)
    
    if not reply:
        reply = "Errore: risposta non riconosciuta dal sistema."
        
    return {"reply": reply}

# === AVVIO SERVER ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_smarTina:app", host=API_HOST, port=API_PORT, reload=API_RELOAD)
