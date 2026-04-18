#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  🌐 SmarTina – Multi-Agente concettuale con memoria GPT e RAG per ITSSocial

"""
Ruoli:
- 🧭 Orchestratore GPT → decide se serve il RAG (informazioni su ITSSocial) o una risposta generica (conversazione libera).
- 📚 Agente RAG GPT → fornisce informazioni dai documenti locali (funzionalità di ITSSocial: home, tendenze, profilo, post, stelle, commenti, accesso, contatti).
- 💬 Agente Generico GPT → gestisce conversazioni spontanee e mantiene la memoria durante la sessione.

Memoria:
- Tutta la conversazione è condivisa tra orchestratore e agenti.
- La memoria è concettuale: GPT comprende il contesto e lo riutilizza, senza variabili esplicite.
"""


# === IMPORTAZIONI ===========================================================

import os, pickle, faiss, numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# === CONFIGURAZIONE AMBIENTE ===============================================

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise SystemExit("❌ Manca la chiave API nel file .env")

client = OpenAI(api_key=api_key)

# Modelli
MODEL_MAIN = "gpt-4o-mini"  # orchestratore
MODEL_FT   = "ft:gpt-4o-mini-2024-07-18:its-cadmo:smartina:CcpM9wrx"
EMBEDDING_MODEL = "text-embedding-3-small"

# File RAG
INDEX_PATH = "rag/its_social_faiss_index.faiss"
METADATA_PATH = "rag/its_social_metadata.pkl"

# === CARICAMENTO BASE DI CONOSCENZA RAG ====================================

index = faiss.read_index(INDEX_PATH)
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

# === FUNZIONI RAG ===========================================================

def get_embedding(text):
    emb = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return emb.data[0].embedding

def cerca_blocchi_simili(query, k=2):
    vec = np.array(get_embedding(query), dtype="float32").reshape(1, -1)
    D,I = index.search(vec, k)
    return [metadata[i] for i in I[0] if i < len(metadata)]

def agente_rag(conversation_history):
    """
    Agente informativo (RAG) che usa la conoscenza dei documenti locali di ITSSocial.
    """
    ultimo_input = conversation_history[-1]["content"]
    blocchi = cerca_blocchi_simili(ultimo_input, k=2)
    contesto = "\n---\n".join(blocchi)
    prompt = [
        {"role": "system", "content": (
            "Sei l'agente informativo di SmarTina, l’assistente ufficiale di ITSSocial. "
            "Usa le informazioni trovate nei documenti RAG per rispondere in modo "
            "chiaro, positivo e coerente con la conversazione. "
            "Parla solo di ITSSocial: home, profilo, post, stelle, tendenze, commenti, regole, accesso e contatti. "
            "Non parlare di bandi, tirocini, docenti o aziende."
        )},
        {"role": "system", "content": f"Contesto utile dai documenti:\n{contesto}"}
    ] + conversation_history

    resp = client.chat.completions.create(model=MODEL_FT, messages=prompt)
    return resp.choices[0].message.content.strip()


# === AGENTE GENERICO =======================================================

def agente_generico(conversation_history):
    """
    Agente generico con memoria concettuale completa.
    Usa la storia della conversazione per rispondere in modo coerente e naturale.
    """
    prompt = [
        {"role": "system", "content": (
            "Sei SmarTina, assistente ufficiale di ITSSocial. "
            "Parla con tono positivo, coinvolgente e amichevole. "
            "Ricorda ciò che l’utente dice nella sessione: nome, interessi e preferenze. "
            "Concentrati sempre su ITSSocial e la sua community di studenti. "
            "Non parlare di tirocini, bandi, aziende o docenti."
        )}
    ] + conversation_history

    resp = client.chat.completions.create(model=MODEL_FT, messages=prompt)
    return resp.choices[0].message.content.strip()

# === ORCHESTRATORE GPT =====================================================

def orchestratore(conversation_history):
    """
    Decide concettualmente a chi passare la richiesta (RAG o Generico),
    basandosi su tutto il contesto della conversazione.
    """
    prompt = [
        {"role": "system", "content": (
            "Sei l'orchestratore di SmarTina. Analizza la conversazione e decidi chi deve rispondere.\n"
            "Se la richiesta riguarda ITSSocial (home, profilo, post, stelle, tendenze, commenti, regole, accesso, contatti) → CALL:RAG:<testo>\n"
            "Se invece è una conversazione generica o personale → CALL:GEN:<testo>\n"
            "Rispondi solo con una di queste due forme, senza aggiungere altro testo."
        )}
    ] + conversation_history


    resp = client.chat.completions.create(model=MODEL_MAIN, messages=prompt)
    return resp.choices[0].message.content.strip()

# === CICLO PRINCIPALE ======================================================

conversation_history = []

print("===============================================")
print("🌐 SmarTina – Multi-Agente Concettuale con Memoria GPT + RAG")
print("Scrivi 'exit' o 'quit' per uscire.")
print("===============================================\n")

while True:
    user_input = input("👤 Tu: ").strip()
    if user_input.lower() in {"exit", "quit"}:
        print("👋 SmarTina ti saluta. Alla prossima!")
        break
    if not user_input:
        continue

    # Aggiungi messaggio utente alla memoria
    conversation_history.append({"role": "user", "content": user_input})

    # 1️⃣ Orchestratore decide concettualmente chi deve rispondere
    decision = orchestratore(conversation_history)

    # 2️⃣ Routing tecnico (basato su CALL)
    if decision.startswith("CALL:RAG:"):
        risposta = agente_rag(conversation_history)
    else:
        risposta = agente_generico(conversation_history)

    # 3️⃣ Aggiorna la memoria con la risposta del bot
    conversation_history.append({"role": "assistant", "content": risposta})

    # 4️⃣ Mostra la risposta
    print(f"SmarTina: {risposta}\n")
