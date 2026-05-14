#!/usr/bin/env python3

# -*- coding: utf-8 -*-

# 💬 SmarTina – Prompt Tuning + Memory (senza orchestratore)



"""

Funzionamento:

- Il modello fine-tuned decide autonomamente se rispondere con INFO o GEN.

- Mantiene memoria del nome utente.

- Knowledge base ITSSocial: Home, Profilo, Post, Tendenze, Contatti, Accesso.

"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# === 1. CONFIGURAZIONE ===
load_dotenv(override=True) # Forza il ricaricamento del file .env

# Cerchiamo la chiave
api_key = os.getenv("SMARTINA_KEY")

# Controllo di sicurezza immediato
if not api_key or len(api_key) < 20:
    print("❌ ERRORE: Chiave API non trovata o troppo corta nel file .env!")
    print("Verifica di avere: SMARTINA_KEY=sk-proj-...")
    exit()

MODEL_FT = "ft:gpt-4o-mini-2024-07-18:its-cadmo:smartina:CcpM9wrx"

# Inizializziamo il modello
try:
    llm = ChatOpenAI(model=MODEL_FT, temperature=0.7, openai_api_key=api_key)
except Exception as e:
    print(f"❌ Errore inizializzazione: {e}")
    exit()

# === 2. KNOWLEDGE BASE ===
# (Ho lasciato solo un esempio, usa quella completa che abbiamo scritto prima)
KNOWLEDGE = {
    "info": "SmarTina assistente ITSSocial. Focus su corsi ITS e piattaforma social. No eventi."
}
knowledge_text_string = "\n".join([f"- {v}" for v in KNOWLEDGE.values()])

# === 3. PROMPT E MEMORIA ===
storia_chat = []
memoria_utente = {"nome": ""}

prompt = ChatPromptTemplate.from_messages([
    ("system", "Sei SmarTina, l'assistente ufficiale di ITSSocial. Non inventare eventi.\n\n"
               "INFO: {knowledge_text}\n\n"
               "UTENTE: {user_info}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | llm | StrOutputParser()

# === 4. LOOP ===
print("✅ Connessione stabilita! SmarTina è pronta.")

while True:
    u_input = input("👤 Tu: ").strip()
    if not u_input or u_input.lower() in ["exit", "quit"]: break

    try:
        # Passiamo correttamente tutte le variabili richieste dal prompt
        risposta = chain.invoke({
            "input": u_input,
            "knowledge_text": knowledge_text_string,
            "user_info": f"Nome: {memoria_utente['nome']}" if memoria_utente["nome"] else "Sconosciuto",
            "history": storia_chat[-6:]
        })
        print(f"💬 SmarTina: {risposta}\n")
        storia_chat.append(HumanMessage(content=u_input))
        storia_chat.append(AIMessage(content=risposta))
    except Exception as e:
        print(f"❌ Errore durante il dialogo: {e}")