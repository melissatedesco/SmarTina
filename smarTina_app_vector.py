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


import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv

# Import LangChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# === 1. CONFIGURAZIONE AMBIENTE ===
load_dotenv(override=True)
api_key = os.getenv("SMARTINA_KEY", "").strip()

if not api_key.startswith("sk-"):
    print("❌ Errore: SMARTINA_KEY non valida nel file .env")
    exit()

MODEL_FT = "ft:gpt-4o-mini-2024-07-18:its-cadmo:smartina:CcpM9wrx"

# Inizializziamo il Modello
llm = ChatOpenAI(model=MODEL_FT, temperature=0.7, openai_api_key=api_key)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)

# === 2. CARICAMENTO RAG (FAISS) ===
INDEX_PATH = "rag/its_social_faiss_index.faiss"
METADATA_PATH = "rag/its_social_metadata.pkl"

try:
    # Carichiamo l'indice FAISS esistente
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    
    # Integriamo FAISS in LangChain per una gestione più semplice
    # Creiamo un wrapper che LangChain può usare
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore({i: metadata[i] for i in range(len(metadata))}),
        index_to_docstore_id={i: i for i in range(len(metadata))}
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    print("✅ Database RAG caricato correttamente.")
except Exception as e:
    print(f"⚠️ Avviso: Impossibile caricare il database RAG ({e}). Lo script userà solo la logica base.")
    retriever = None

# === 3. DEFINIZIONE PROMPT ===
# Qui uniamo tutte le istruzioni: No Eventi + Sì Classi/Video + Memoria
prompt = ChatPromptTemplate.from_messages([
    ("system", "Sei SmarTina, l'assistente ufficiale di ITSSocial. "
               "REGOLE:\n"
               "1. Usa il CONTESTO fornito dai documenti per rispondere.\n"
               "2. Se l'utente chiede di docenti o video, spiega le Classi Aperte (YouTube) e Chiuse.\n"
               "3. NON parlare mai di eventi, workshop o date future.\n"
               "4. Sii amichevole e chiama l'utente per nome se lo conosci.\n\n"
               "CONTESTO RAG:\n{context}\n\n"
               "INFO UTENTE: {user_info}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | llm | StrOutputParser()

# === 4. LOGICA DI SUPPORTO ===
storia_chat = []
memoria_utente = {"nome": ""}

# === 5. LOOP PRINCIPALE ===
print("\n===============================================")
print("🌐 SmarTina LangChain RAG (Expert Edition)")
print("Scrivi 'exit' per uscire.")
print("===============================================\n")

while True:
    u_input = input("👤 Tu: ").strip()
    if not u_input: continue
    if u_input.lower() in ["exit", "quit"]: break

    if u_input.lower() == "dimentica tutto":
        storia_chat = []
        memoria_utente["nome"] = ""
        print("🧽 Memoria resettata!\n")
        continue

    # Gestione Nome
    if u_input.lower().startswith(("mi chiamo ", "il mio nome è ")):
        nome = u_input.split()[-1].strip().capitalize()
        memoria_utente["nome"] = nome
        print(f"💬 SmarTina: Piacere {nome}! Me lo sono segnato. 😊\n")
        continue

    # 1️⃣ Recupero Contesto dal RAG (se disponibile)
    context_text = ""
    if retriever:
        docs = retriever.invoke(u_input)
        context_text = "\n\n".join([d.page_content for d in docs])
    
    # 2️⃣ Aggiunta info manuali per Classi/Video (se non presenti nel RAG)
    if any(k in u_input.lower() for k in ["classe", "video", "prof"]):
        context_text += "\nDidattica: Esistono Classi Chiuse (private) e Classi Aperte (video YouTube)."

    user_info = f"L'utente si chiama {memoria_utente['nome']}." if memoria_utente["nome"] else "Nome sconosciuto."

    # 3️⃣ Generazione Risposta
    try:
        risposta = chain.invoke({
            "input": u_input,
            "context": context_text,
            "user_info": user_info,
            "history": storia_chat[-6:]
        })

        print(f"💬 SmarTina: {risposta}\n")

        # Aggiornamento Storia
        storia_chat.append(HumanMessage(content=u_input))
        storia_chat.append(AIMessage(content=risposta))

    except Exception as e:
        print(f"❌ Errore: {e}")