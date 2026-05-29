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
CHUNK_SIZE    = 100
CHUNK_OVERLAP = 200
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
    # 1. Rimuoviamo i link web temporaneamente per non farli corrompere da TextBlob/Speller
    # Trova tutto ciò che inizia con http o https
    urls = re.findall(r'https?://\S+', text)
    text_clean = re.sub(r'https?://\S+', '[URL_PLACEHOLDER]', text)

    # 2. CORREZIONE ORTOGRAFICA (Richiesta del progetto)
    # Autocorrect corregge i refusi digitati male
    text_clean = speller(text_clean)
    # TextBlob fa un secondo controllo sulla grammatica
    text_clean = str(TextBlob(text_clean).correct())

    # 3. NLP CON SPACY (Rimozione Stop Words controllata per non rompere il contesto)
    # Creiamo il documento spaCy
    doc = nlp(text_clean)
    
    # Teniamo le parole importanti, ma togliamo i simboli strani o spazi di troppo.
    # Filtriamo i token che sono punteggiatura pura,
    # ma manteniamo la struttura della frase per l'embedding di OpenAI.
    tokens = [token.text for token in doc if not token.is_space]
    text_clean = " ".join(tokens)

    # 4. Ripristiniamo i link originali al loro posto
    for url in urls:
        text_clean = text_clean.replace('[URL_PLACEHOLDER]', url, 1)

    # 5. Pulizia finale degli spazi
    return re.sub(r"\s+", " ", text_clean).strip()


# === 5. GENERAZIONE EMBEDDING ===
def get_embeddings(texts: list) -> np.ndarray:
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vectors  = [item.embedding for item in response.data]
    return np.array(vectors, dtype="float32")


# === 6. SALVATAGGIO INDICE FAISS ===
def build_and_save(metadata: list, vectors: np.ndarray, index_dir: str):
    os.makedirs(index_dir, exist_ok=True)

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

# Salviamo l'indice vettoriale
    faiss.write_index(index, os.path.join(index_dir, "its_social_faiss_index.faiss"))

  # Salviamo la METADATA per l'orchestratore  
    with open(os.path.join(index_dir, "its_social_metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ Indice salvato in: {index_dir}")


def main():
    if len(sys.argv) < 2:
        raise SystemExit("❌ Uso: python rag/create_vector_store.py <smarTina_contenuti.txt>")

    path = sys.argv[1]

    # Caricamento
    raw_text = load_document(path)

   # Creiamo la METADATA (i chunk originali)
    metadata = split_text(raw_text)
    if not metadata:
        raise SystemExit("❌ Nessun chunk generato. Controlla il file sorgente.")
    print(f"✂️  {len(metadata)} chunk creati e salvati come metadata")

    # Pre-processing per gli embedding (i chunk originali vengono preservati)
    print("🔧 Pre-processing in corso...")
    processed_chunks = [preprocess_text(c) for c in metadata]

    # Embedding dei chunk processati
    print("🔢 Generazione embedding...")
    vectors = get_embeddings(processed_chunks)

    # Salvataggio: indice FAISS (vettori processati) + metadata (testi originali)
    print("💾 Salvataggio indice FAISS e metadata...")
    build_and_save(metadata, vectors, INDEX_DIR)

    print(f"\n📚 Chunk indicizzati: {len(metadata)}")
    print("🌟 SmarTina è pronta a rispondere su ITSocial!")


if __name__ == "__main__":
    main()
