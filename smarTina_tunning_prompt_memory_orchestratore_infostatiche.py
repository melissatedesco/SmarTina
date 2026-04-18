#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 💬 SmarTina – Assistente ITSSocial con Orchestratore e Memoria

"""
Funzionalità:
- Orchestratore GPT decide se una richiesta è INFO (knowledge base) o GEN (chiacchiera)
- Agente INFO risponde con informazioni statiche su ITSSocial
- Agente GENERICO gestisce conversazioni libere, usando la memoria del nome utente
- Comandi speciali:
    - "Mi chiamo <nome>" → salva nome
    - "Cosa ricordi?" / "Cosa sai di me" → mostra memoria
    - "Dimentica tutto" → resetta memoria
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# === CONFIGURAZIONE ========================================================

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise SystemExit("❌ Manca la chiave API nel file .env")

client = OpenAI(api_key=api_key)

MODEL_MAIN = "gpt-4o-mini"  # orchestratore
MODEL_FT   = "ft:gpt-4o-mini-2024-07-18:its-cadmo:smartina:CcpM9wrx"

# === KNOWLEDGE BASE ========================================================
INFO = {
    "home": "Nella Home di ITSSocial puoi vedere i post pubblicati dagli studenti, commentare e mettere le stelle ai contenuti che ti piacciono di più.",
    "profilo": "Nel Profilo puoi visualizzare le tue informazioni personali e i post che hai pubblicato.",
    "post": "Su ITSSocial puoi pubblicare post per condividere ciò che stai facendo, i tuoi lavori o le tue idee.",
    "tendenze": "La sezione Tendenze mostra i post che hanno ricevuto più stelle.",
    "contatti": "Per assistenza o informazioni puoi contattare il team di ITSSocial tramite email: socialitsinfo@gmail.com."
}

# === MEMORIA TEMPORANEA ===================================================
memoria = {"nome_utente": ""}
conversation_history = []

MAX_HISTORY = 10
def add_history(role, content):
    conversation_history.append({"role": role, "content": content})
    if len(conversation_history) > MAX_HISTORY:
        conversation_history.pop(0)

# === ORCHESTRATORE =========================================================
def orchestratore(user_input):
    prompt = [
        {"role": "system", "content": (
            "Sei l'orchestratore di SmarTina, assistente ITSSocial. "
            "Decidi se la richiesta riguarda informazioni del social "
            "(Home, Profilo, Post, Tendenze, Contatti, Accesso) o una chiacchiera.\n"
            "Se riguarda il social → CALL:INFO:<testo>\n"
            "Altrimenti → CALL:GEN:<testo>\n"
            "Rispondi solo in questa forma, senza aggiungere altro."
        )},
        {"role": "user", "content": user_input}
    ]
    resp = client.chat.completions.create(model=MODEL_MAIN, messages=prompt)
    return resp.choices[0].message.content.strip()

# === AGENTE INFO ===========================================================
def agente_info(user_input):
    prompt = [
        {"role": "system", "content": (
            "Hai accesso alle informazioni di ITSSocial:\n"
            f"Home: {INFO['home']}\n"
            f"Profilo: {INFO['profilo']}\n"
            f"Post: {INFO['post']}\n"
            f"Tendenze: {INFO['tendenze']}\n"
            f"Contatti: {INFO['contatti']}\n"
            f"Accesso: {INFO['accesso']}\n\n"
            "Rispondi in modo chiaro, gentile e conciso. Non inventare dati non presenti."
        )},
        {"role": "user", "content": user_input}
    ]
    resp = client.chat.completions.create(model=MODEL_FT, messages=prompt)
    return resp.choices[0].message.content.strip()

# === AGENTE GENERICO =======================================================
def agente_generico(user_input, memoria=None, history=None):
    messages = []

    # Prompt semplice con nome utente e nome assistente
    system_content = "Tu sei SmarTina, l'assistente virtuale di ITSSocial. "
    if memoria and memoria.get("nome_utente"):
        system_content += f"L'utente si chiama {memoria['nome_utente']}. Usa sempre il suo nome quando appropriato. "
    system_content += "Chiama te stessa SmarTina, mai Assistant."

    messages.append({"role": "system", "content": system_content})

    # Cronologia chat
    if history:
        for h in history[-MAX_HISTORY:]:
            messages.append({"role": h["role"], "content": h["content"]})

    messages.append({"role": "user", "content": user_input})

    resp = client.chat.completions.create(model=MODEL_FT, messages=messages)
    return resp.choices[0].message.content.strip()

# === CICLO PRINCIPALE ======================================================
print("===============================================")
print("💬 SmarTina – Assistente ITSSocial con Orchestratore")
print("Scrivi 'exit' per uscire.")
print("===============================================\n")

while True:
    user_input = input("👤 Tu: ").strip()
    if user_input.lower() in {"exit", "quit"}:
        print("👋 SmarTina ti saluta. Alla prossima!")
        break
    if not user_input:
        continue

    # --- Memorizza nome utente in modo semplice ---
    if user_input.lower().startswith("mi chiamo"):
        nome = user_input[9:].strip().capitalize()  # tutto dopo "mi chiamo"
        memoria["nome_utente"] = nome
        print(f"💬 SmarTina: Piacere, {nome}! Ora lo ricorderò.\n")
        continue

    if user_input.lower().startswith("il mio nome è"):
        nome = user_input[13:].strip().capitalize()  # tutto dopo "il mio nome è"
        memoria["nome_utente"] = nome
        print(f"💬 SmarTina: Piacere, {nome}! Ora lo ricorderò.\n")
        continue

    # --- Mostra memoria ---
    if user_input.lower() in {"cosa ricordi", "cosa sai di me"}:
        if memoria["nome_utente"]:
            print(f"💬 SmarTina: Ricordo che ti chiami {memoria['nome_utente']} 💡\n")
        else:
            print("💬 SmarTina: Non ho ancora memorizzato il tuo nome. 😊\n")
        continue

    # --- Dimentica tutto ---
    if user_input.lower() == "dimentica tutto":
        memoria["nome_utente"] = ""
        conversation_history.clear()
        print("🧽 SmarTina: Memoria cancellata!\n")
        continue

    # --- Orchestratore ---
    decision = orchestratore(user_input)

    if decision.startswith("CALL:INFO:"):
        query = decision.replace("CALL:INFO:", "").strip()
        risposta = agente_info(query)
    else:
        query = decision.replace("CALL:GEN:", "").strip()
        risposta = agente_generico(query, memoria, conversation_history)

    # --- Aggiorna cronologia ---
    add_history("user", user_input)
    add_history("assistant", risposta)

    print(f"💬 SmarTina: {risposta}\n")