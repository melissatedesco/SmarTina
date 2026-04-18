#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 💬 SmarTina – Assistente ITSSocial con con Orchestratore, Memoria e Ticket

"""
RUOLI:
- 🧭 Orchestratore GPT → decide se la richiesta è INFO (knowledge base) o GEN (chiacchiera)
- ℹ️ Agente INFO GPT → risponde con informazioni statiche su ITSSocial
- 💬 Agente GENERICO GPT → gestisce conversazioni libere, usando la memoria del nome utente e la cronologia
- 📝 Ticket → permette di aprire ticket in chat (Assistenza Tecnica o Supporto Admin)
"""


import os
from dotenv import load_dotenv
from openai import OpenAI
from smarTina_app_vector_ticket_db_api import (
    get_db,
    mostra_ticket_da_db,
    registra_ticket
)


# === CONFIGURAZIONE =========================================================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("❌ Manca la chiave API nel file .env")

client = OpenAI(api_key=api_key)

MODEL_MAIN = "gpt-4o-mini"
MODEL_FT   = "ft:gpt-4o-mini-2024-07-18:its-cadmo:smartina:CcpM9wrx"

# File RAG
# INDEX_PATH = "rag/its_social_faiss_index.faiss"
# METADATA_PATH = "rag/its_social_metadata.pkl"
# === MEMORIA ===============================================================
memoria = {"nome_utente": ""}
conversation_history = []
MAX_HISTORY = 10

# === FUNZIONI ==============================================================
def add_history(role, content):
    conversation_history.append({"role": role, "content": content})
    if len(conversation_history) > MAX_HISTORY:
        conversation_history.pop(0)

# === TICKET SYSTEM =========================================================
def apri_ticket(titolo, tipo):
    """Crea un ticket e lo salva nel database."""
    nome = memoria.get("nome_utente", "").strip()
    cognome = memoria.get("cognome_utente", "").strip()
    username = memoria.get("username", "").strip()
    email = memoria.get("email", "").strip()

    # Se manca anche solo 1 dato → non permetto l'inserimento
    if not (nome and cognome and username and email):
        return None
    
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO ticket (nome_utente, cognome_utente, username, email, tipo_ticket, descrizione) VALUES (%s, %s, %s, %s, %s, %s)",
        )
        conn.commit()
    finally:
        cur.close()
        conn.close()

    return {"titolo": titolo, "tipo": tipo, "nome": nome}

# === ORCHESTRATORE =========================================================
def orchestratore(user_input):
    prompt = [
        {"role": "system", "content": (
            "Sei l'orchestratore di SmarTina. Decidi se la richiesta è INFO (knowledge base) o GEN (conversazione). "
            "Se è INFO → CALL:INFO:<testo>, altrimenti → CALL:GEN:<testo>."
        )},
        {"role": "user", "content": user_input}
    ]
    resp = client.chat.completions.create(model=MODEL_MAIN, messages=prompt)
    return resp.choices[0].message.content.strip()

# === AGENTE INFO ===========================================================
def agente_info(user_input):
    user_lower = user_input.lower()
    if "ticket" in user_lower and any(p in user_lower for p in ["vedere", "visualizzare", "lista", "mostrare"]):
        return "I ticket si gestiscono solo qui in chat con me 😊 Puoi scrivere 'lista ticket' per vedere quelli già aperti."
    prompt = [
        {"role": "system", "content": (
            "Rispondi in modo chiaro e conciso, usando solo informazioni verificate su ITSSocial."
        )},
        {"role": "user", "content": user_input}
    ]
    resp = client.chat.completions.create(model=MODEL_FT, messages=prompt)
    return resp.choices[0].message.content.strip()

# === AGENTE GENERICO =======================================================
def agente_generico(user_input, memoria=None, history=None):
    messages = []
    system_content = "Tu sei SmarTina, assistente virtuale di ITSSocial. "
    if memoria.get("nome_utente"):
        system_content += f"Ricorda che l'utente si chiama {memoria['nome_utente']}. "
    system_content += "Rispondi sempre come SmarTina."

    messages.append({"role": "system", "content": system_content})
    if history:
        for h in history[-10:]:
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_input})

    resp = client.chat.completions.create(model=MODEL_FT, messages=messages)
    return resp.choices[0].message.content.strip()

# === CICLO PRINCIPALE ======================================================
print("===============================================")
print("💬 SmarTina – Assistente ITSSocial con Ticket e Memoria Temporanea")
print("===============================================\n")

while True:
    user_input = input("👤 Tu: ").strip()

    if not user_input:
        continue

    # Comando uscita
    if user_input.lower() in {"exit", "quit"}:
        print("👋 SmarTina ti saluta. Alla prossima!")
        break

    # === COMANDI MEMORIA ====================================================
    if user_input.lower().startswith(("mi chiamo", "il mio nome è")):
        memoria["nome_utente"] = user_input.split()[-1].capitalize()
        print(f"💬 SmarTina: Piacere, {memoria['nome_utente']}! Ora lo ricorderò.\n")
        continue

    if user_input.lower().startswith("il mio cognome è"):
        memoria["cognome_utente"] = user_input.split()[-1].capitalize()
        print(f"💬 SmarTina: Ok, ricorderò che il tuo cognome è {memoria['cognome_utente']}!\n")
        continue

    if user_input.lower().startswith("il mio username è"):
        memoria["username"] = user_input.split()[-1]
        print(f"💬 SmarTina: Perfetto! Il tuo username è {memoria['username']}.\n")
        continue

    if user_input.lower().startswith("la mia email è"):
        memoria["email"] = user_input.split()[-1]
        print(f"💬 SmarTina: Ho salvato la tua email {memoria['email']} 📧\n")
        continue
         
    if user_input.lower() in {"cosa ricordi", "cosa sai di me"}:
        if memoria["nome_utente"]:
            print(f"💬 SmarTina: Ricordo che ti chiami {memoria['nome_utente']} 💡\n")
        else:
            print("💬 SmarTina: Non ho ancora memorizzato il tuo nome. 😊\n")
        continue

    if user_input.lower() == "dimentica tutto":
        memoria["nome_utente"] = ""
        conversation_history.clear()
        print("🧽 SmarTina: Memoria cancellata!\n")
        continue

    # === COMANDI TICKET =====================================================
    if any(frase in user_input.lower() for frase in ["vedere i ticket", "lista ticket", "ticket aperti"]):
        print("💬 Ecco i ticket registrati nel database:")
        print(mostra_ticket_da_db() + "\n")
        continue

    elif any(frase in user_input.lower() for frase in [
        "fare un ticket", "aprire un ticket", "voglio fare un ticket", "ho bisogno di fare un ticket"
    ]):
        print("💬 Vuoi aprire un nuovo ticket adesso? (si/no)")
        conferma = input("👤 ").strip().lower()

        if conferma == "si":
            if not memoria.get("nome_utente"):
                print("❌ Devi prima dire il tuo nome. Scrivi 'Mi chiamo [nome]' prima di aprire un ticket.\n")
                continue

            titolo = input("💬 Scrivi il titolo del ticket: ").strip()
            if not titolo:
                print("\033[91m💬 Devi inserire un titolo valido.\033[0m\n")
                continue

            print("💬 Che tipo di ticket vuoi aprire?")
            print("  1 → Assistenza Tecnica")
            print("  2 → Supporto Admin")
            scelta = input("👤 Inserisci 1 o 2: ").strip()
            tipo = "Supporto Admin" if scelta == "2" else "Assistenza Tecnica"

            ticket = apri_ticket(titolo, tipo)
            if ticket:
                print(f"✅ Ticket salvato nel database: {ticket['titolo']} ({ticket['tipo']})\n")
            else:
                print("❌ Errore: il ticket non è stato salvato perché manca il nome utente.\n")

        elif conferma == "no":
            print("💬 Va bene, non apriamo il ticket per ora.\n")
        continue

    # === ORCHESTRATORE ======================================================
    decision = orchestratore(user_input)
    if decision.startswith("CALL:INFO:"):
        query = decision.replace("CALL:INFO:", "").strip()
        risposta = agente_info(query)
    else:
        query = decision.replace("CALL:GEN:", "").strip()
        risposta = agente_generico(query, memoria, conversation_history)

    add_history("user", user_input)
    add_history("assistant", risposta)
    print(f"💬 SmarTina: {risposta}\n")
