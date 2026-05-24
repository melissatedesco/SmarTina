#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 💬 SmarTina – Assistente ITSocial con Orchestratore e Memoria

"""
Funzionalità:
- Orchestratore GPT decide se una richiesta è INFO (knowledge base) o GEN (chiacchiera)
- Agente INFO risponde con informazioni statiche su ITSocial
- Agente GENERICO gestisce conversazioni libere, usando la memoria del nome utente
- Comandi speciali per la gestione della memoria locale.
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

# === KNOWLEDGE BASE AGGIORNATA =============================================
INFO = {
    "home": "Nella Home di ITSocial puoi vedere i post pubblicati dagli studenti, commentare e mettere le stelle ai contenuti che ti piacciono di più.",
    "profilo": "Nel Profilo puoi visualizzare le tue informazioni personali e i post che hai pubblicato.",
    "post": "Su ITSocial puoi pubblicare post per condividere ciò che stai facendo, i tuoi lavori o le tue idee.",
    "tendenze": "La sezione Tendenze mostra i post che hanno ricevuto più stelle.",
    "contatti": "Per assistenza o informazioni puoi contattare il team di ITSocial tramite email: socialitsinfo@gmail.com.",
    "accesso": "Puoi accedere a ITSocial con le tue credenziali studente oppure registrarti dalla pagina principale.",
    "its_academy": "Gli ITS Academy sono percorsi di specializzazione tecnica post-diploma della durata di 2 anni (circa 1800-2000 ore). Almeno il 35% del tempo viene svolto in azienda e oltre il 50% dei docenti proviene dal mondo del lavoro. Rilasciano il Diploma Tecnico Superiore (EQF livello 5) riconosciuto in tutta l'Unione Europea e sono suddivisi in 6 aree tecnologiche: Efficienza energetica, Mobilità sostenibile, Nuove tecnologie della vita, Made in Italy, Tecnologie del turismo e ICT.",
    "ticket": "Per aprire un ticket basta scrivere in chat 'Voglio aprire un ticket' o contattare l'assistenza via email a socialitsinfo@gmail.com. Le tipologie disponibili sono: Richiesta informazioni, Segnalazione problema, Supporto tecnico e Feedback sull'esperienza."
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
            "Sei l'orchestratore di SmarTina. Il tuo unico compito è smistare la richiesta dell'utente.\n\n"
            "Devi rispondere RIGIDAMENTE con CALL:INFO:<testo> se l'utente chiede informazioni su:\n"
            "- Funzionalità del social (Home, Profilo, Post, Tendenze, Accesso)\n"
            "- Assistenza, contatti o apertura Ticket (Contatti, Ticket)\n"
            "- Informazioni, definizioni o spiegazioni su cosa sono gli 'ITS', gli 'Academy' o gli 'ITS Academy'.\n\n"
            "Devi rispondere RIGIDAMENTE con CALL:GEN:<testo> SOLO se l'utente fa richieste generiche, "
            "saluti (ciao, come va), complimenti o provocazioni/frasi fuori tema non legate ai punti sopra citati.\n\n"
            "Rispondi solo ed esclusivamente in questa forma, senza aggiungere spazi vuoti o altro."
        )},
        {"role": "user", "content": user_input}
    ]
    resp = client.chat.completions.create(model=MODEL_MAIN, messages=prompt, timeout=10.0)
    return resp.choices[0].message.content.strip()

# === AGENTE INFO ===========================================================
def agente_info(user_input):
    prompt = [
        {"role": "system", "content": (
            "Sei l'agente informativo di SmarTina. Sii sempre gentile, educata e professionale.\n"
            "Rispondi alle domande usando ESCLUSIVAMENTE i dati testuali forniti qui sotto. Non inventare nulla.\n"
            "Se l'utente ti chiede degli 'ITS', delle 'Academy' o degli 'ITS Academy', usa la voce 'Its Academy'.\n\n"
            f"Home: {INFO['home']}\n"
            f"Profilo: {INFO['profilo']}\n"
            f"Post: {INFO['post']}\n"
            f"Tendenze: {INFO['tendenze']}\n"
            f"Contatti: {INFO['contatti']}\n"
            f"Accesso: {INFO['accesso']}\n"
            f"Its Academy: {INFO['its_academy']}\n"
            f"Ticket: {INFO['ticket']}\n"
        )},
        {"role": "user", "content": user_input}
    ]
    resp = client.chat.completions.create(model=MODEL_FT, messages=prompt, timeout=10.0)
    return resp.choices[0].message.content.strip()

# === AGENTE GENERICO =======================================================
def agente_generico(user_input, memoria=None, history=None):
    messages = []

   # Configurazione della personalità gentile di SmarTina
    system_content = (
        "Tu sei SmarTina, l'assistente virtuale di ITSocial. "
        "Il tuo tratto fondamentale è la GENTILEZZA: sii sempre estremamente educata, calma, pacifica e professionale. "
        "Non offendere, non usare mai parole volgari e non insultare MAI l'utente, "
    )
    
    if memoria and memoria.get("nome_utente"):
        system_content += f"L'utente si chiama {memoria['nome_utente']}. Usa il suo nome quando appropriato per essere amichevole. "
    
    system_content += "Chiama te stessa SmarTina, mai Assistant."

    messages.append({"role": "system", "content": system_content})

    if history:
        for h in history[-MAX_HISTORY:]:
            messages.append({"role": h["role"], "content": h["content"]})

    messages.append({"role": "user", "content": user_input})

    resp = client.chat.completions.create(model=MODEL_FT, messages=messages, timeout=10.0)
    return resp.choices[0].message.content.strip()

# === CICLO PRINCIPALE ======================================================
print("===============================================")
print("💬 SmarTina – Assistente ITSocial con Orchestratore")
print("Scrivi 'exit' per uscire.")
print("===============================================\n")

while True:
    user_input = input("👤 Tu: ").strip()
    if user_input.lower() in {"exit", "quit"}:
        print("👋 SmarTina ti saluta. Alla prossima!")
        break
    if not user_input:
        continue

    # --- 1. VERIFICA DELLA MEMORIA  ---
    if any(domanda in user_input.lower() for domanda in {"come mi chiamo", "ti ricordi come mi chiamo", "cosa ricordi", "cosa sai di me"}):
        if memoria["nome_utente"]:
            print(f"💬 SmarTina: Ti chiami {memoria['nome_utente']}! 😊\n")
        else:
            print("💬 SmarTina: Non mi hai ancora detto il tuo nome! Come ti chiami? 😊\n")
        continue

    # --- Memorizza nome utente in modo semplice ---
    if "mi chiamo" in user_input.lower():
        idx = user_input.lower().find("mi chiamo") + len("mi chiamo")
        nome = user_input[9:].strip().capitalize()  
        memoria["nome_utente"] = nome
        print(f"💬 SmarTina: Piacere, {nome}! Ora lo ricorderò.\n")
        continue

    if "il mio nome è" in user_input.lower():
        idx = user_input.lower().find("il mio nome è") + len("il mio nome è")
        nome = user_input[idx:].strip(".,! ").capitalize()
        memoria["nome_utente"] = nome
        print(f"💬 SmarTina: Piacere, {nome}! Ora lo ricorderò. 😊\n")
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
    try:
        # Chiamata all'orchestratore
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

    except Exception as e:
        print(f"❌ Ops! C'è stato un problema di connessione con OpenAI.")
        print(f"Dettaglio errore: {e}\n")