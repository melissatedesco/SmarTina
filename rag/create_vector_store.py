#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 🌐 SmarTina – Pipeline RAG con pre-processing avanzato per ITSocial
#
# Uso: python rag/create_vector_store.py data/smarTina_contenuti.txt
#
# Dipendenze: pip install openai python-dotenv faiss-cpu numpy
#             pip install autocorrect textblob spacy
#             python -m spacy download it_core_news_sm

import os
import sys
import re
import pickle
import numpy as np
import faiss
from autocorrect import Speller
from textblob import TextBlob
import spacy
from openai import OpenAI
from dotenv import load_dotenv

# === 1. CONFIGURAZIONE ===
load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY", "").strip()

if not api_key.startswith("sk-"):
    raise SystemExit("❌ OPENAI_API_KEY non valida nel file .env")

client  = OpenAI(api_key=api_key)
nlp     = spacy.load("it_core_news_sm")
speller = Speller(lang="it")

INDEX_DIR     = "rag/its_social_faiss"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
EMBED_MODEL   = "text-embedding-3-small"


# === 2. CARICAMENTO DOCUMENTO ===
def load_document(path: str) -> str:
    if not path.endswith(".txt"):
        raise ValueError(f"❌ Formato non supportato. Usa un file .txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File non trovato: {path}")
    print("📄 Caricamento TXT...")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    print(f"   → {len(text)} caratteri caricati")
    return text


# === 3. DIVISIONE IN CHUNK ===
def split_text(text: str) -> list:
    chunks = []
    start  = 0
    while start < len(text):
        chunk = text[start : start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# === 4. PRE-PROCESSING ===
def preprocess_text(text: str) -> str:
    # Correzione ortografica con autocorrect
    corrected = speller(text)

    # Ulteriore correzione con TextBlob
    corrected = str(TextBlob(corrected).correct())

    # Pulizia: mantieni solo lettere, numeri e spazi
    corrected = re.sub(r"[^\w\s]", " ", corrected)
    corrected = re.sub(r"\s+", " ", corrected).strip()

    # Rimozione stop words italiane con SpaCy
    doc    = nlp(corrected)
    tokens = [t.text for t in doc if not t.is_stop and not t.is_space and t.text]

    return " ".join(tokens)


# === 5. GENERAZIONE EMBEDDING ===
def get_embeddings(texts: list) -> np.ndarray:
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vectors  = [item.embedding for item in response.data]
    return np.array(vectors, dtype="float32")


# === 6. SALVATAGGIO INDICE FAISS ===
def build_and_save(original_chunks: list, vectors: np.ndarray, index_dir: str):
    os.makedirs(index_dir, exist_ok=True)

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(original_chunks, f)

    print(f"✅ Indice salvato in: {index_dir}")


def main():
    if len(sys.argv) < 2:
        raise SystemExit("❌ Uso: python rag/create_vector_store.py <smarTina_contenuti.txt>")

    path = sys.argv[1]

    # Caricamento
    raw_text = load_document(path)

    # Chunking sul testo originale
    original_chunks = split_text(raw_text)
    if not original_chunks:
        raise SystemExit("❌ Nessun chunk generato. Controlla il file sorgente.")
    print(f"✂️  {len(original_chunks)} chunk creati")

    # Pre-processing per gli embedding (i chunk originali vengono preservati)
    print("🔧 Pre-processing in corso...")
    processed_chunks = [preprocess_text(c) for c in original_chunks]

    # Embedding dei chunk processati
    print("🔢 Generazione embedding...")
    vectors = get_embeddings(processed_chunks)

    # Salvataggio: indice FAISS (vettori processati) + metadata (testi originali)
    print("💾 Salvataggio indice FAISS e metadata...")
    build_and_save(original_chunks, vectors, INDEX_DIR)

    print(f"\n📚 Chunk indicizzati: {len(original_chunks)}")
    print("🌟 SmarTina è pronta a rispondere su ITSocial!")


if __name__ == "__main__":
    main()
