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
load_dotenv(override=True)
api_key = os.getenv("SMARTINA_KEY", "").strip()

if not api_key.startswith("sk-"):
    print("❌ Errore: Chiave SMARTINA_KEY non trovata nel file .env")
    exit()

# Modello Fine-Tuned
MODEL_FT = "ft:gpt-4o-mini-2024-07-18:its-cadmo:smartina:CcpM9wrx"

llm = ChatOpenAI(
    model=MODEL_FT, 
    temperature=0.7, 
    openai_api_key=api_key
)

# === 2. KNOWLEDGE BASE (Dati aggiornati) ===
KNOWLEDGE = {
    "its_cadmo": (
        "L'ITS CADMO ha sede a Soverato (CZ) ed è il punto di riferimento per l'ICT in Calabria. "
        "Corsi attivi: Data Analyst & AI Specialist, Software Developer, Digital Media Designer, "
        "Digital & Energy Process Specialist, Cybersecurity Expert. Sito: https://www.itscadmo.it/"
    ),
    "social": (
        "ITSSocial è la piattaforma per gli studenti ITS. "
        "Funzioni: Home (post e stelle), Profilo, Tendenze. "
        "Contatti: socialitsinfo@gmail.com."
    ),
    "didattica": (
        "Sul Social esistono aree didattiche gestite dai professori:\n"
        "- CLASSI CHIUSE: Riservate alla classe del professore per materiali e comunicazioni private.\n"
        "- CLASSI APERTE: Dove i professori caricano video didattici (es. YouTube) focalizzati "
        "sulla programmazione e le competenze tecniche."
    ),
    "calabria": (
        "Esistono diversi ITS in Calabria (Efficienza Energetica, Pegasus, Tirreno, Pinta, M.A.SK., Iridea, Elaia), "
        "ma l'ITS CADMO è l'eccellenza per il settore digitale e informatico."
    )
}

# === 3. ROUTER DI CONTESTO ===
def seleziona_contesto(u_input):
    u = u_input.lower()
    contesto = ""
    
    if any(k in u for k in ["cadmo", "soverato", "informatica", "corsi"]): 
        contesto += KNOWLEDGE["its_cadmo"] + "\n"
    
    if any(k in u for k in ["social", "post", "stelle", "profilo", "tendenze"]): 
        contesto += KNOWLEDGE["social"] + "\n"
        
    if any(k in u for k in ["classe", "prof", "video", "youtube", "programmazione"]): 
        contesto += KNOWLEDGE["didattica"] + "\n"

    if any(k in u for k in ["calabria", "altri its", "elenco"]): 
        contesto += KNOWLEDGE["calabria"] + "\n"
    
    return contesto if contesto else "Rispondi in modo amichevole e professionale."

# === 4. PROMPT E MEMORIA ===
storia_chat = []
memoria_utente = {"nome": ""}

prompt = ChatPromptTemplate.from_messages([
    ("system", "Sei SmarTina, l'assistente ufficiale di ITSSocial. "
               "REGOLE RIGIDE:\n"
               "1. NON parlare di eventi, workshop o calendari.\n"
               "2. Usa solo il CONTESTO fornito per info tecniche.\n"
               "3. Se l'utente chiede di professori o video, spiega le Classi Aperte e Chiuse.\n"
               "4. Dai sempre priorità all'ITS CADMO.\n\n"
               "CONTESTO:\n{context}\n\n"
               "INFO UTENTE: {user_info}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | llm | StrOutputParser()

# === 5. LOOP PRINCIPALE ===
print("===============================================")
print("🚀 SmarTina EXPERT (LangChain + Didattica)")
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

    # Salvataggio Nome
    if u_input.lower().startswith(("mi chiamo ", "il mio nome è ")):
        nome = u_input.split()[-1].strip().replace("?", "").capitalize()
        memoria_utente["nome"] = nome
        print(f"💬 SmarTina: Piacere di conoscerti, {nome}! Ho salvato il tuo nome. 😊\n")
        continue

    # Recupero contesto e risposta
    current_context = seleziona_contesto(u_input)
    user_info = f"L'utente si chiama {memoria_utente['nome']}." if memoria_utente["nome"] else "Nome sconosciuto."
    
    try:
        risposta = chain.invoke({
            "input": u_input,
            "context": current_context,
            "user_info": user_info,
            "history": storia_chat[-6:] # Ricorda gli ultimi 3 scambi (6 messaggi)
        })
        
        # Aggiornamento storia
        storia_chat.append(HumanMessage(content=u_input))
        storia_chat.append(AIMessage(content=risposta))
        
        print(f"💬 SmarTina: {risposta}\n")
    except Exception as e:
        print(f"❌ Errore durante la risposta: {e}")