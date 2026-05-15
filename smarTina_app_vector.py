#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 🌐 SmarTina – Multi-Agente con memoria GPT e RAG per ITSocial

"""
Ruoli:
- 📚 Agente RAG → fornisce informazioni dai documenti locali (ITSocial + ITS Academy).
- 💬 Memoria sessione → mantiene la storia della conversazione e il nome utente.
"""

import os
import pickle
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

# === 1. CONFIGURAZIONE AMBIENTE ===
load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY", "").strip()

if not api_key.startswith("sk-"):
    print("❌ Errore: OPENAI_API_KEY non valida nel file .env")
    exit()

client = OpenAI(api_key=api_key)

MODEL_FT    = "ft:gpt-4o-mini-2024-07-18:its-cadmo:smartina:CcpM9wrx"
EMBED_MODEL = "text-embedding-3-small"
MAX_HISTORY = 6

# === 2. CARICAMENTO RAG (FAISS) ===
INDEX_DIR = "rag/its_social_faiss"

try:
    index = faiss.read_index(os.path.join(INDEX_DIR, "index.faiss"))
    with open(os.path.join(INDEX_DIR, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)
    print("✅ Database RAG caricato correttamente.")
except Exception as e:
    print(f"⚠️ Impossibile caricare il database RAG ({e}).")
    print("   Esegui prima: python rag/create_vector_store.py")
    index  = None
    chunks = []

# === 3. FUNZIONE DI RETRIEVAL ===
def retrieve_context(query: str, k: int = 3) -> str:
    if index is None:
        return ""
    response = client.embeddings.create(model=EMBED_MODEL, input=[query])
    vec      = np.array([response.data[0].embedding], dtype="float32")
    _, ids   = index.search(vec, k)
    return "\n\n".join(chunks[i] for i in ids[0] if i < len(chunks))

# === 4. LOGICA DI SUPPORTO ===
storia_chat    = []
memoria_utente = {"nome": ""}

# === 5. LOOP PRINCIPALE ===
print("\n===============================================")
print("🌐 SmarTina RAG")
print("Scrivi 'exit' per uscire.")
print("===============================================\n")

while True:
    u_input = input("👤 Tu: ").strip()
    if not u_input:
        continue
    if u_input.lower() in ["exit", "quit"]:
        break

    if u_input.lower() == "dimentica tutto":
        storia_chat = []
        memoria_utente["nome"] = ""
        print("🧽 Memoria resettata!\n")
        continue

    if u_input.lower().startswith(("mi chiamo ", "il mio nome è ")):
        nome = u_input.split()[-1].strip().capitalize()
        memoria_utente["nome"] = nome
        print(f"💬 SmarTina: Piacere {nome}! Me lo sono segnato. 😊\n")
        continue

    # 1️⃣ Recupero Contesto dal RAG
    context_text = retrieve_context(u_input)

    user_info = (
        f"L'utente si chiama {memoria_utente['nome']}."
        if memoria_utente["nome"] else "Nome sconosciuto."
    )

    # 2️⃣ Generazione Risposta
    system_prompt = (
        "Sei SmarTina, l'assistente ufficiale di ITSocial.\n"
        "REGOLE:\n"
        "1. Usa il CONTESTO fornito dai documenti per rispondere.\n"
        "2. Se l'utente chiede di classi, spiega la differenza tra classi pubbliche e private.\n"
        "3. NON inventare informazioni non presenti nel contesto.\n"
        "4. Sii gentile e chiama l'utente per nome se lo conosci.\n\n"
        f"CONTESTO RAG:\n{context_text}\n\n"
        f"INFO UTENTE: {user_info}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages += storia_chat[-MAX_HISTORY:]
    messages.append({"role": "user", "content": u_input})

    try:
        resp     = client.chat.completions.create(model=MODEL_FT, messages=messages)
        risposta = resp.choices[0].message.content.strip()

        print(f"💬 SmarTina: {risposta}\n")

        storia_chat.append({"role": "user",      "content": u_input})
        storia_chat.append({"role": "assistant", "content": risposta})

    except Exception as e:
        print(f"❌ Errore: {e}")
