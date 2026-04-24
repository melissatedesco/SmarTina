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
from langchain.agents import create_openai_functions_agent
from langchain.agents.agent import AgentExecutor

# === 1. CONFIGURAZIONE ===
load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise SystemExit("❌ Manca la chiave API nel file .env")

# Utilizziamo il tuo modello Fine-Tuned
MODEL_FT = "ft:gpt-4o-mini-2024-07-18:its-cadmo:smartina:CcpM9wrx"
llm = ChatOpenAI(model=MODEL_FT, temperature=0.7, openai_api_key=api_key)

# === 2. KNOWLEDGE BASE (DATABASE) ===
KNOWLEDGE = {
    "its_cadmo": (
        "L'ITS CADMO ha sede a Soverato (CZ) ed è focalizzato sull'ICT (Informatica). "
        "Corso biennale di 2000 ore con 800 ore di tirocinio. Sito: https://www.itscadmo.it/"
    ),
    "calabria": (
        "ITS Academy in Calabria:\n"
        "- ITS Cadmo (Soverato): ICT\n"
        "- Its Efficienza Energetica (Reggio Calabria): Energia\n"
        "- Its Pegasus (Polistena): Mobilità Sostenibile\n"
        "- Its Tirreno (Fuscaldo): Chimica e Nuove Tecnologie della Vita\n"
        "- Its Pinta (Cotronei): Agroalimentare\n"
        "- Its M.A.SK. (San Ferdinando): Servizi alle Imprese\n"
        "- Its Iridea (Cosenza): Agroalimentare\n"
        "- Its Elaia Calabria (Vibo Valentia): Turismo e Beni Culturali"
    ),
    "social": (
        "ITSSocial è per TUTTI gli studenti ITS d'Italia. "
        "Funzioni: Home (post e stelle), Profilo, Tendenze e Contatti (socialitsinfo@gmail.com). "
        "Novità: Sezione didattica dove i prof caricano link a video (es. JavaScript) consigliati."
    )
}

# === 3. ROUTER LOGICO (Context Selector) ===
def seleziona_contesto(u_input):
    u = u_input.lower()
    contesto = ""
    if any(k in u for k in ["cadmo", "iscrizione", "soverato"]): contesto += KNOWLEDGE["its_cadmo"] + "\n"
    if any(k in u for k in ["calabria", "elenco", "quali sono", "sede"]): contesto += KNOWLEDGE["calabria"] + "\n"
    if any(k in u for k in ["social", "piattaforma", "video", "stelle", "post"]): contesto += KNOWLEDGE["social"] + "\n"
    return contesto if contesto else "Sii amichevole e rispondi come SmarTina."

# === 4. PROMPT E MEMORIA ===
storia_chat = []
memoria_utente = {"nome": ""}

prompt = ChatPromptTemplate.from_messages([
    ("system", "Sei SmarTina, l'assistente ufficiale di ITSSocial e degli ITS.\n"
               "Usa queste info se necessario: {context}\n"
               "Info Utente: {user_info}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# Creazione della Chain
chain = prompt | llm | StrOutputParser()

# === 5. LOOP PRINCIPALE ===
print("🚀 SmarTina LangChain Online (Calabria & Social Edition)")
print("-------------------------------------------------------")

while True:
    u_input = input("👤 Tu: ").strip()
    if not u_input: continue
    if u_input.lower() in ["exit", "quit"]: break
    
    # Dimentica tutto
    if u_input.lower() == "dimentica tutto":
        storia_chat = []
        memoria_utente["nome"] = ""
        print("🧽 Memoria pulita!\n")
        continue

    # Gestione Nome (Logica manuale per efficienza)
    if u_input.lower().startswith(("mi chiamo", "il mio nome è")):
        nome = u_input.split()[-1].strip().capitalize()
        memoria_utente["nome"] = nome
        print(f"💬 SmarTina: Piacere {nome}! Lo ricorderò. 😊\n")
        continue

    # 1. Recupero contesto basato sulla domanda
    current_context = seleziona_contesto(u_input)
    
    # 2. Generazione risposta via LangChain
    try:
        risposta = chain.invoke({
            "input": u_input,
            "context": current_context,
            "user_info": f"L'utente si chiama {memoria_utente['nome']}" if memoria_utente["nome"] else "Nome sconosciuto.",
            "history": storia_chat[-6:] # Ultime 3 coppie di messaggi
        })
        
        # 3. Aggiornamento Storia
        storia_chat.append(HumanMessage(content=u_input))
        storia_chat.append(AIMessage(content=risposta))
        
        print(f"💬 SmarTina: {risposta}\n")
    except Exception as e:
        print(f"❌ Errore: {e}")