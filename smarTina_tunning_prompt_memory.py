#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 💬 SmarTina – Prompt Tuning + Memory (senza orchestratore)

"""
Funzionamento:
- Il modello fine-tuned decide autonomamente se rispondere con INFO o GEN.
- Mantiene memoria del nome utente.
- Knowledge base ITSocial: tutte le funzionalità della piattaforma.
"""

import os
import re
from openai import OpenAI
from dotenv import load_dotenv

# === CONFIGURAZIONE ========================================================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("❌ Manca la chiave API nel file .env")

client = OpenAI(api_key=api_key)

MODEL_FT    = "ft:gpt-4o-mini-2024-07-18:its-cadmo:smartina:CcpM9wrx"
MAX_HISTORY = 10

INFO = {
    "cos_e":      "ITSocial è il social network per studenti e professori degli ITS Academy italiani. Unisce funzioni social (post, like, commenti, messaggi, follow) e scolastiche (classi, compiti, materiali, annunci).",
    "ruoli":      "4 ruoli: Studente (social + classi in lettura + consegna compiti), Professore (social + crea/gestisce classi), Istituto (gestione professori), Admin (accesso completo).",
    "app_mobile": "Disponibile come web e app iOS/Android. Stesse credenziali. Include feed, messaggi, classi, notifiche, profilo e chat SmarTina.",
    "accesso":    "Registrazione: /registrazione con email valida. Login: /login. Token sessione: 1 ora (non è un bug). Recupero password: /recupera-password → email → codice → nuova password.",
    "home_feed":  "Home a 3 colonne: menu sinistra, feed centro, Tendenze destra. 4 tab feed: Tutti (10 post a scroll), Seguiti, Annunci, Notifiche.",
    "post":       "Composer in cima al feed. Max 500 caratteri, 5 allegati, sondaggio fino a 5 opzioni. Post non modificabili: elimina e ripubblica. Elimina solo i propri post.",
    "like":       "Icona stella sotto il post per mettere/togliere like. Funziona anche dai Tendenze.",
    "salvataggio":"Icona segnalibro per salvare/rimuovere. Post salvati nel profilo → tab Salvati.",
    "commenti":   "Icona commento sotto il post. I propri commenti sono modificabili ed eliminabili. Non si possono modificare commenti altrui.",
    "profilo":    "Mostra nome, username, foto, bio (max 160 char), contatori post/follower/seguiti. Tab: Post, Salvati, Mi piace. Foto tramite URL online (es. Imgur). Modifica con icona matita.",
    "ricerca":    "Barra di ricerca in home, risultati in tempo reale su nome e username. Cliccando si visita il profilo o si apre una chat.",
    "messaggi":   "Apri chat da profilo utente o sezione Messaggi. Funzioni: Rispondi, Fissa (24h/7gg/30gg), Importante (stella gialla), Inoltra. Pallino verde = online ultimi 2 minuti. Polling automatico.",
    "notifiche":  "8 tipi: like, commenti, nuovi follower, richiesta iscrizione classe, iscrizione approvata/rifiutata, nuovo annuncio, messaggi. Aggiornamento ogni 8 secondi. Solo in-app, nessuna email.",
    "classi":     "PUBBLICA: visibile in Esplora, iscrizione automatica. PRIVATA: serve codice 8 caratteri, attesa approvazione. 4 sezioni: Bacheca, Lavori, Persone, Materiali. Compiti: Ok/>3gg, Urgent/≤3gg, Over/scaduto.",
    "ticket":     "Scrivi 'Voglio aprire un ticket' o contatta socialitsinfo@gmail.com. Tipi: Richiesta info, Segnalazione problema, Supporto tecnico, Feedback.",
    "its_academy":"ITS Academy: percorsi post-diploma, 2 anni ~1800-2000 ore, ≥35% in azienda, ≥50% docenti dal lavoro. Titolo: Diploma Tecnico Superiore (EQF 5). Riconosciuto UE. 6 aree: energia, mobilità, vita, Made in Italy, turismo, ICT.",
    "contatti":   "Email supporto: socialitsinfo@gmail.com",
}

# === MEMORIA ===============================================================
memoria = {
    "nome_utente": "",
    "storia":      [],
}

def mem_add(role, content):
    memoria["storia"].append({"role": role, "content": content})
    if len(memoria["storia"]) > MAX_HISTORY:
        memoria["storia"].pop(0)

# === CICLO PRINCIPALE ======================================================
print("===============================================")
print("💬 SmarTina – Prompt Tuning + Memory")
print("===============================================\n")

while True:
    user_input = input("👤 Tu: ").strip()

    if not user_input:
        continue
    if user_input.lower() in {"exit", "quit"}:
        print("👋 SmarTina ti saluta. Alla prossima!")
        break

    # === MEMORIA NOME UTENTE ===============================================
    # Rileva il nome ovunque nel messaggio (es. "ciao mi chiamo X", "mi chiamo X tu?")
    _match = re.search(r"(?:mi chiamo|il mio nome è)\s+([A-Za-zÀ-ÿ]+)", user_input, re.IGNORECASE)
    if _match:
        memoria["nome_utente"] = _match.group(1).capitalize()
    # Il messaggio passa sempre all'LLM: risponde al saluto E a eventuali domande

    if user_input.lower() in {"cosa ricordi", "cosa sai di me"}:
        if memoria["nome_utente"]:
            print(f"💬 SmarTina: Ricordo che ti chiami {memoria['nome_utente']} 💡\n")
        else:
            print("💬 SmarTina: Non ho ancora memorizzato il tuo nome. 😊\n")
        continue

    if user_input.lower() == "dimentica tutto":
        memoria["nome_utente"] = ""
        memoria["storia"].clear()
        print("🧽 SmarTina: Memoria cancellata!\n")
        continue

    # === COSTRUZIONE PROMPT ================================================
    info_text = "\n".join([f"{k.title()}: {v}" for k, v in INFO.items()])
    nome      = memoria["nome_utente"]
    nome_rule = (
        f"REGOLA ASSOLUTA: l'utente si chiama {nome}. Usa il nome {nome} nella risposta. "
        "Non usare mai altri nomi."
    ) if nome else ""

    messages = [
        {"role": "system", "content": (
            f"Sei SmarTina, assistente ufficiale di ITSocial.\n"
            f"{nome_rule}\n"
            f"Informazioni disponibili:\n{info_text}\n\n"
            "Rispondi solo con informazioni presenti qui sopra. "
            "Non inventare nomi di ITS, eventi o dati non presenti. "
            "Rispondi in modo chiaro, gentile e conciso."
        )}
    ]

    messages += memoria["storia"][-5:]
    messages.append({"role": "user", "content": user_input})

    resp   = client.chat.completions.create(model=MODEL_FT, messages=messages)
    answer = resp.choices[0].message.content.strip()

    mem_add("user", user_input)
    mem_add("assistant", answer)

    print(f"💬 SmarTina: {answer}\n")
