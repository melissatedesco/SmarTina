#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 💬 SmarTina – Prompt Tuning + Memory (senza orchestratore)
"""
Funzionamento:
- Il modello fine-tuned decide autonomamente se rispondere con INFO o GEN.
- Mantiene memoria del nome utente.
- Knowledge base ITSocial: Home, Profilo, Post, Tendenze, Contatti, Accesso, Its_Academy, Ticket.
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

# 💡 Configurato con gpt-4o-mini per i test PRE fine-tuning
MODEL_FT = "gpt-4o-mini-2024-07-18:its-cadmo:"
MAX_HISTORY = 10

INFO = {
    "home": "Nella Home di ITSocial puoi vedere i post pubblicati dagli studenti, commentare e mettere le stelle ai contenuti che ti piacciono di più.",
    "profilo": "Nel Profilo puoi visualizzare le tue informazioni personali e i post che hai pubblicato.",
    "post": "Su ITSocial puoi pubblicare post per condividere ciò che stai facendo, i tuoi lavori o le tuoi idee.",
    "tendenze": "La sezione Tendenze mostra i post che hanno ricevuto più stelle.",
    "contatti": "Per assistenza o informazioni puoi contattare il team di ITSocial tramite email: socialitsinfo@gmail.com",
    "accesso": "Puoi accedere a ITSocial con le tue credenziali studente oppure registrarti dalla pagina principale.",
    "its_academy": "Gli ITS Academy sono percorsi di specializzazione tecnica post-diploma della durata di 2 anni (circa 1800-2000 ore). Almeno il 35% del tempo viene svolto in azienda e oltre il 50% dei docenti proviene dal mondo del lavoro. Rilasciano il Diploma Tecnico Superiore (EQF livello 5) riconosciuto in tutta l'Unione Europea e sono suddivisi in 6 aree tecnologiche: Efficienza energetica, Mobilità sostenibile, Nuove tecnologie della vita, Made in Italy, Tecnologie del turismo e ICT.",
    "ticket": "Per aprire un ticket basta scrivere in chat 'Voglio aprire un ticket' o contattare l'assistenza via email a socialitsinfo@gmail.com. Le tipologie disponibili sono: Richiesta informazioni, Segnalazione problema, Supporto tecnico e Feedback sull'esperienza."
}

# === MEMORIA ===============================================================
memoria = {
    "nome_utente": "",
    "storia": []
}

def mem_add(role, content):
    memoria["storia"].append({"role": role, "content": content})
    if len(memoria["storia"]) > MAX_HISTORY:
        memoria["storia"].pop(0)

# === CICLO PRINCIPALE ======================================================
print("===============================================")
print("💬 SmarTina – Prompt Tuning + Memory (autonoma)")
print("===============================================\n")

while True:
    user_input = input("👤 Tu: ").strip()

    if not user_input:
        continue
    if user_input.lower() in {"exit", "quit"}:
        print("👋 SmarTina ti saluta. Alla prossima!")
        break

    # === MEMORIA NOME UTENTE ===============================================
    if user_input.lower().startswith(("mi chiamo", "il mio nome è")):
        nome = user_input.split(maxsplit=2)[-1].strip().capitalize()
        memoria["nome_utente"] = nome
        print(f"💬 SmarTina: Piacere, {nome}! Ora lo ricorderò.\n")
        continue

    # Mostra memoria
    if user_input.lower() in {"cosa ricordi", "cosa sai di me"}:
        if memoria["nome_utente"]:
            print(f"💬 SmarTina: Ricordo che ti chiami {memoria['nome_utente']} 💡\n")
        else:
            print("💬 SmarTina: Non ho ancora memorizzato il tuo nome. 😊\n")
        continue

    # Dimentica tutto
    if user_input.lower() == "dimentica tutto":
        memoria["nome_utente"] = ""
        memoria["storia"].clear()
        print("🧽 SmarTina: Memoria cancellata!\n")
        continue

    # === COSTRUZIONE PROMPT ================================================
    knowledge_text = "\n".join([f"{k.title().replace('_', ' ')}: {v}" for k, v in INFO.items()])
    nome_txt = f"L'utente si chiama {memoria['nome_utente']}." if memoria["nome_utente"] else ""

    messages = [
        {"role": "system", "content": (
            f"Sei SmarTina, assistente ufficiale di ITSocial. "
            f"Hai accesso a queste informazioni:\n{knowledge_text}\n"
            f"{nome_txt}\n"
            "Decidi autonomamente se la richiesta dell'utente riguarda informazioni del social "
            "(Home, Profilo, Post, Tendenze, Contatti, Accesso, Its Academy, Ticket) o è generica. "
            "Rispondi in modo chiaro, gentile e conciso."
        )}
    ]

    messages += memoria["storia"][-5:]
    messages.append({"role": "user", "content": user_input})

    try:
        resp = client.chat.completions.create(model=MODEL_FT, messages=messages)
        answer = resp.choices[0].message.content.strip()

        mem_add("user", user_input)
        mem_add("assistant", answer)

        print(f"💬 SmarTina: {answer}\n")
    except Exception as e:
        print(f"❌ Errore API: {e}\n")