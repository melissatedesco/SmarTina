#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 🌐 SmarTina – Assistente ufficiale di ITSSocial (Multi-Agente con Memoria e RAG)

"""
Ruoli:
- 🧭 Orchestratore GPT → decide se la richiesta riguarda ITSSocial (RAG) o è generica (GEN).
- ℹ️ Agente INFO GPT → risponde con informazioni su home, profilo, post, stelle, tendenze, commenti, accesso, contatti.
- 💬 Agente Generico GPT → gestisce conversazioni libere e mantiene la memoria della sessione.



Questo script crea anche l’indice FAISS e i metadata per il RAG di ITSSocial.
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
EMBEDDING_MODEL = "text-embedding-3-small"

# File RAG
INDEX_PATH = "rag/its_social_faiss_index.faiss"
METADATA_PATH = "rag/its_social_metadata.pkl"

# === DATI BASE ==========================================================
# Puoi aggiungere qui testi, FAQ, descrizioni o regolamenti di ITSSocial
documenti = [
    "ITSSocial è la piattaforma social dedicata agli studenti degli ITS italiani.",
    "Nella sezione Home puoi vedere i post pubblicati dagli studenti e interagire con la community.",
    "Nel Profilo puoi aggiornare le tue informazioni personali e visualizzare i tuoi post.",
    "La sezione Tendenze mostra i contenuti più apprezzati, con il maggior numero di stelle.",
    "Per assistenza puoi contattare il team ITSSocial all'indirizzo support@itssocial.it.",
    "Puoi accedere a ITSSocial con le credenziali del tuo ITS. Se non hai un account, puoi registrarti dal sito ufficiale.",
]

# === CREAZIONE EMBEDDING ================================================
print("🔄 Generazione embedding per i contenuti ITSSocial...")
embeddings = []
for doc in documenti:
    emb = client.embeddings.create(model=EMBEDDING_MODEL, input=doc)
    embeddings.append(emb.data[0].embedding)

X = np.array(embeddings, dtype="float32")

# === CREAZIONE INDICE FAISS =============================================
print("📦 Creazione indice FAISS...")
index = faiss.IndexFlatL2(X.shape[1])
index.add(X)

# ✅ Assicuriamoci che la cartella esista
os.makedirs("rag", exist_ok=True)

# === SALVATAGGIO METADATA ===============================================
faiss.write_index(index, INDEX_PATH)
with open(METADATA_PATH, "wb") as f:
    pickle.dump(documenti, f)


# === RISULTATO ============================================================

print("\n✅ Indice FAISS e metadata creati con successo!")
print(f"📁 File indice: {INDEX_PATH}")
print(f"📁 File metadata: {METADATA_PATH}")
print("\n🌟 SmarTina è pronta a rispondere su ITSSocial!")