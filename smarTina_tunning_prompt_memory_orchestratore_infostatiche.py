#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 💬 SmarTina – Assistente ITSocial con Orchestratore e Memoria

"""
Funzionalità:
- Orchestratore GPT decide se la richiesta è INFO (knowledge base) o GEN (chiacchiera)
- Agente INFO risponde con informazioni statiche su ITSocial
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

MODEL_MAIN  = "gpt-4o-mini"
MODEL_FT    = "ft:gpt-4o-mini-2024-07-18:its-cadmo:smartina:CcpM9wrx"
MAX_HISTORY = 10

# === KNOWLEDGE BASE ========================================================
KNOWLEDGE = {
    "cos_e":      "ITSocial è il social network per studenti e professori degli ITS Academy italiani. Unisce funzioni social (post, like, commenti, messaggi, follow) e scolastiche (classi, compiti, materiali, annunci).",
    "ruoli":      "4 ruoli: Studente (social + classi in lettura + consegna compiti), Professore (social + crea/gestisce classi), Istituto (gestione professori).",
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
    "classi":     "Solo i Professori hanno il pulsante 'Crea Classe'. Lo studente può accedere solo alle classi dei propri professori o dell'istituto di appartenenza tramite codice a 8 caratteri. Non è possibile iscriversi a classi di altre scuole o percorsi non pertinenti. PUBBLICA: visibile in Esplora, iscrizione automatica. PRIVATA: non visibile, serve codice, iscrizione in attesa di approvazione. 4 sezioni: Bacheca, Lavori, Persone, Materiali. Compiti: Ok/>3gg, Urgent/≤3gg, Over/scaduto.",
    "netiquette": "ITSocial è un ambiente professionale e scolastico. Sono vietati post off-topic, selfie o foto personali non inerenti alla didattica. I contenuti devono essere pertinenti al percorso ITS e al contesto accademico.",
    "ticket":     "Scrivi 'Voglio aprire un ticket' o contatta socialitsinfo@gmail.com. Tipi: Richiesta info, Segnalazione problema, Supporto tecnico, Feedback.",
    "its_academy":"Gli ITS Academy non sono solo informatica. Esistono 6 aree tecnologiche nazionali: Efficienza Energetica, Mobilità Sostenibile, Nuove Tecnologie della Vita, Made in Italy (agroalimentare, meccanica, moda), Turismo e Beni Culturali, ICT. Percorsi post-diploma, 2 anni ~1800-2000 ore, ≥35% in azienda, ≥50% docenti dal mondo del lavoro. Titolo: Diploma Tecnico Superiore (EQF 5), riconosciuto in tutta l'UE.",
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

# === ORCHESTRATORE =========================================================
def orchestratore(user_input):
    prompt = [
        {"role": "system", "content": (
            "Sei l'orchestratore di SmarTina, assistente ITSocial. "
            "Decidi se la richiesta riguarda informazioni della piattaforma "
            "(cos'è, ruoli, accesso, home, post, like, salvataggio, commenti, profilo, "
            "ricerca, messaggi, notifiche, classi, netiquette, ticket, ITS Academy, contatti) o è una chiacchiera generica.\n"
            "Se riguarda la piattaforma → CALL:INFO:<testo>\n"
            "Altrimenti → CALL:GEN:<testo>\n"
            "Rispondi solo in questa forma, senza aggiungere altro."
        )},
        {"role": "user", "content": user_input},
    ]
    resp = client.chat.completions.create(model=MODEL_MAIN, messages=prompt)
    return resp.choices[0].message.content.strip()

# === AGENTE INFO ===========================================================
def agente_info(user_input):
    info_text = "\n".join([f"{k.title()}: {v}" for k, v in KNOWLEDGE.items()])
    nome      = memoria["nome_utente"]
    nome_rule = (
        f"REGOLA ASSOLUTA: l'utente si chiama {nome}. "
        f"Usa il nome {nome} nella risposta. Non usare mai altri nomi."
    ) if nome else "Non è stato fornito un nome utente."

    prompt = [
        {"role": "system", "content": (
            f"Sei SmarTina, assistente ufficiale di ITSocial.\n"
            f"{nome_rule}\n\n"
            f"Informazioni disponibili:\n{info_text}\n\n"
            "Rispondi in modo chiaro, gentile e conciso. Non inventare dati non presenti."
        )},
        {"role": "user", "content": user_input},
    ]
    resp = client.chat.completions.create(model=MODEL_FT, messages=prompt)
    return resp.choices[0].message.content.strip()

# === AGENTE GENERICO =======================================================
def agente_generico(user_input):
    nome = memoria["nome_utente"]
    nome_rule = (
        f"REGOLA ASSOLUTA: l'utente si chiama {nome}. "
        f"Usa il nome {nome} quando è naturale farlo. Non usare mai altri nomi."
    ) if nome else "Il nome dell'utente non è ancora noto."

    system_content = (
        "Sei SmarTina, l'assistente virtuale di ITSocial. Chiama te stessa SmarTina, mai Assistant.\n"
        f"{nome_rule}"
    )

    messages = [{"role": "system", "content": system_content}]
    messages += memoria["storia"][-5:]
    messages.append({"role": "user", "content": user_input})

    resp = client.chat.completions.create(model=MODEL_FT, messages=messages)
    return resp.choices[0].message.content.strip()

# === CICLO PRINCIPALE ======================================================
print("===============================================")
print("💬 SmarTina – Assistente ITSocial con Orchestratore")
print("Scrivi 'exit' per uscire.")
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

    # === ORCHESTRATORE =====================================================
    decision = orchestratore(user_input)

    if decision.startswith("CALL:INFO:"):
        query    = decision.replace("CALL:INFO:", "").strip()
        risposta = agente_info(query)
    else:
        query    = decision.replace("CALL:GEN:", "").strip()
        risposta = agente_generico(query)

    mem_add("user", user_input)
    mem_add("assistant", risposta)

    print(f"💬 SmarTina: {risposta}\n")
