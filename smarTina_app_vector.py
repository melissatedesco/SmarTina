#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 🌐 SmarTina – Multi-Agente con memoria GPT e RAG per ITSocial

"""
Ruoli:
- 🧭 Orchestratore GPT → decide se serve il RAG (informazioni su ITSocial o ITS regionali) o una risposta generica.
- 📚 Agente RAG GPT → fornisce informazioni dai documenti locali (funzionalità di ITSocial e database ITS regionali).
- 💬 Agente Generico GPT → gestisce conversazioni spontanee e mantiene la memoria durante la sessione.

Memoria:
- Tutta la conversazione è condivisa tra orchestratore e agenti.
- La memoria è concettuale: GPT comprende il contesto e lo riutilizza, senza variabili esplicite.
"""

import os
import pickle
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

# === CONFIGURAZIONE AMBIENTE ===============================================

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise SystemExit("❌ Manca la chiave API nel file .env")

client = OpenAI(api_key=api_key)

# Modelli
MODEL_MAIN = "gpt-4o-mini"  # orchestratore
MODEL_FT   = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# File RAG (Verifica che i nomi dei file generati corrispondano a questi)
INDEX_PATH = "rag/its_social_faiss/its_social_faiss_index.faiss"
METADATA_PATH = "rag/its_social_faiss/its_social_metadata.pkl"

# === CARICAMENTO BASE DI CONOSCENZA RAG ====================================

try:
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    print("✅ Database RAG caricato con successo!")
except Exception as e:
    print(f"⚠️ Errore nel caricamento dei file RAG: {e}")
    print("Assicurati di aver lanciato prima lo script di creazione del vector store.")
    index = None
    metadata = []

# === FUNZIONI RAG ===========================================================

def get_embedding(text):
    emb = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return emb.data[0].embedding

def cerca_blocchi_simili(query, k=2):
    if index is None or len(metadata) == 0:
        return []
    vec = np.array(get_embedding(query), dtype="float32").reshape(1, -1)
    D, I = index.search(vec, k)
    return [metadata[i] for i in I[0] if i < len(metadata)]

def agente_rag(conversation_history):
    """
    Agente informativo (RAG) che usa la conoscenza dei documenti locali di ITSocial.
    """
    ultimo_input = conversation_history[-1]["content"]
    blocchi = cerca_blocchi_simili(ultimo_input, k=5)
    contesto = "\n---\n".join(blocchi)
    prompt = [
        {"role": "system", "content": (
            "Sei l'agente informativo di SmarTina, l’assistente ufficiale di ITSocial. "
            "Usa le informazioni trovate nei documenti RAG per rispondere in modo "
            "chiaro, positivo e coerente con la conversazione. "
            "Usa le informazioni per spiegare le funzionalità del sito (home, messaggi, classi pubbliche/private) "
            "oppure per fornire i dettagli e i link dei vari ITS Academy regionali richiesti dall'utente. "
            "Non inventare dati o link se non sono esplicitamente scritti nel contesto."
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
            "Sei SmarTina, assistente ufficiale di ITSocial. "
            "Parla con tono positivo, coinvolgente e amichevole. "
            "Ricorda ciò che l’utente dice nella sessione: nome, interessi e preferenze. "
            "Concentrati sempre su ITSocial e la sua community di studenti. "
            "Non parlare di tirocini, bandi, aziende o docenti esterni."
        )}
    ] + conversation_history

    resp = client.chat.completions.create(model=MODEL_FT, messages=prompt)
    return resp.choices[0].message.content.strip()

# === ORCHESTRATORE GPT AGGIORNATO ===========================================

def orchestratore(conversation_history):
    """
    Analizza la storia della conversazione e smista intelligentemente la richiesta 
    all'agente RAG (per dati tecnici/regionali) o all'agente Generico (per chiacchiere).
    """
    prompt = [
        {"role": "system", "content": (
            "Sei l'orchestratore di SmarTina. Analizza l'intera conversazione e decidi chi deve rispondere.\n\n"
            "Attiva il modulo RAG se la richiesta dell'utente riguarda:\n"
            "1. Funzionalità specifiche di ITSocial (home, profilo, post, stelle, tendenze, commenti, regole, accesso, classi virtuali, contatti, ticket).\n"
            "2. Informazioni generali sugli ITS Academy (definizione, aree tecnologiche, livelli EQF, durata, diploma).\n"
            "3. Richieste su specifiche regioni italiane (es. 'Quali ITS ci sono in Calabria?', 'Dammi i link della Campania', ecc.).\n"
            "4. Il nome di un ITS specifico (es. 'ITS Cadmo', 'ITS Pegasus', 'Fondazione Pinta', qualsiasi nome di istituto).\n"
            "5. Qualsiasi domanda che inizia con 'cosa è', 'dimmi di', 'vorrei sapere', 'informazioni su' riguardo a ITS o ITSocial.\n"
            "--> In questo caso rispondi SEMPRE e soltanto con: CALL:RAG\n\n"
            "Attiva il modulo GENERICO SOLO se l'utente sta facendo:\n"
            "1. Saluti o congedi puri (es. 'ciao', 'buongiorno', 'grazie', 'arrivederci').\n"
            "2. Chiacchiere completamente fuori tema, sta dicendo il proprio nome o domande personali/emotive senza riferimento a ITS.\n"
            "--> In questo caso rispondi SEMPRE e soltanto con: CALL:GEN\n\n"
            "REGOLA CRITICA: In caso di dubbio scegli sempre CALL:RAG. Rispondi esclusivamente con 'CALL:RAG' o 'CALL:GEN'."
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

    # 1️⃣ Orchestratore decide chi deve rispondere
    decision = orchestratore(conversation_history)

    # 2️⃣ Routing tecnico (Reso più robusto con il controllo parziale 'in')
    if "CALL:RAG" in decision:
        risposta = agente_rag(conversation_history)
    else:
        risposta = agente_generico(conversation_history)

    # 3️⃣ Aggiorna la memoria con la risposta del bot
    conversation_history.append({"role": "assistant", "content": risposta})

    # 4️⃣ Mostra la risposta
    print(f"💬 SmarTina: {risposta}\n")