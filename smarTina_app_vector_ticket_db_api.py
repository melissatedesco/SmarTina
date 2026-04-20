#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# smarTina_app_vector_ticket_db.py

import os
import re
import pickle
import faiss
import numpy as np
from typing import List, Dict
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import mysql.connector
import json
# ciao
# === CONFIG ===
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("Manca OPENAI_API_KEY in .env")

client = OpenAI(api_key=api_key)

MODEL_MAIN = os.getenv("MODEL_MAIN", "gpt-4o-mini")
MODEL_FT = os.getenv("MODEL_FT", "ft:gpt-4o-mini-2024-07-18:its-cadmo:smartina:CcpM9wrx")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

INDEX_PATH = os.getenv("INDEX_PATH")
METADATA_PATH = os.getenv("METADATA_PATH", "rag/itssocial_metadata.pkl")

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT")),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),
    "database": os.getenv("DB_NAME"),
}

# === UTILS ===

def extract_json_block(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(json)?\s*|\s*```$", "", s, flags=re.IGNORECASE)
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("JSON non trovato nella risposta del modello.")
    return s[start:end+1]

def validate_and_normalize_payload(raw_json: str):
    """Convalida e normalizza i dati del ticket."""
    payload_json = extract_json_block(raw_json)
    data = json.loads(payload_json)

    required = ["nome_utente", "cognome_utente", "username", "email", "tipo_ticket"]

    missing = [k for k in required if k not in data or not str(data[k]).strip()]
    if missing:
        raise ValueError(f"Mancano i campi: {', '.join(missing)}")

    return {
        "nome_utente": data["nome_utente"].strip(),
        "cognome_utente": data["cognome_utente"].strip(),
        "username": data["username"].strip(),
        "email": data["email"].strip(),
        "tipo_ticket": data["tipo_ticket"].strip(),
    }

# === DB ===

def get_db():
    return mysql.connector.connect(**DB_CONFIG)
    
def registra_ticket(nome, cognome, username, email, tipo):
    """Registra un nuovo ticket nel database con data/ora automatica."""
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO ticket 
            (nome_utente, cognome_utente, username, email_utente, tipo_ticket)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (nome, cognome, username, email, tipo)
        )
        conn.commit()
    finally:
        cur.close()
        conn.close()

def mostra_ticket_da_db():
    """Legge e mostra tutti i ticket presenti nel database."""
    conn = get_db()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT nome_utente, cognome_utente, username, email_utente, tipo_ticket, created_at FROM ticket ORDER BY id DESC")
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    if not rows:
        return "Nessun ticket trovato."

    testo = "Lista ticket nel database:\n"
    for r in rows:
        testo += f"- {r['nome_utente']} | {r['cognome_utente']} | {r['username']} | {r['email_utente']} | {r['tipo_ticket']} | [ORA: {r['created_at']}]\n"
    return testo

# === RAG ===

try:
    index = faiss.read_index(INDEX_PATH)
except Exception:
    index = None

try:
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
except Exception:
    metadata = []

def get_embedding(t):
    emb = client.embeddings.create(model=EMBEDDING_MODEL, input=t)
    return emb.data[0].embedding

def cerca_blocchi_simili(q, k=2):
    """Cerca blocchi simili nel RAG; se RAG non disponibile, ritorna lista vuota."""
    if index is None or not metadata:
        return []
    v = np.array(get_embedding(q), dtype="float32").reshape(1, -1)
    D, I = index.search(v, k)
    return [metadata[i] for i in I[0] if i < len(metadata) and i != -1]

# === AGENTI ===

def orchestratore(hist: list) -> str:
    try:
        prompt = [
            {"role": "system",
             "content": (
                 "Sei l'orchestratore di SmarTina. "
                 "Analizza la conversazione e decidi chi deve rispondere.\n"
                 "Se riguarda un ticket → CALL:TICKET:<testo>\n"
                 "Se riguarda informazioni o contenuti → CALL:RAG:<testo>\n"
                 "Altrimenti → CALL:GEN:<testo>\n"
                 "Rispondi solo con una di queste forme."
             )}
        ] + hist
        r = client.chat.completions.create(model=MODEL_MAIN, messages=prompt)
        return r.choices[0].message.content.strip()
    except Exception as e:
        print(" ERRORE orchestratore:", e)
        return "CALL:GEN:Errore interno, procedi con risposta generica."

def agente_ticket(hist):
    p = [
        {"role": "system",
         "content": "Richiedi dati: nome_utente,cognome_utente, username, email, tipo_ticket. "
                    "Quando tutti i dati sono presenti rispondi SOLO con CALL:CONFIRMED:{...JSON...}. "
                    "Tipi ticket disponibili:\n"
                    "- Richiesta informazioni\n"
                    "- Segnalazione problema\n"
                    "- Supporto tecnico\n"
                    "- Feedback sull’esperienza"}
    ] + hist
    r = client.chat.completions.create(model=MODEL_MAIN, messages=p)
    return r.choices[0].message.content.strip()

def agente_rag(hist):
    try:
        ultimo = hist[-1]["content"]
        blocchi = cerca_blocchi_simili(ultimo, 2)
        if not blocchi:
            return "Al momento non ho informazioni specifiche su questa richiesta."

        p = [
            {"role": "system", "content": "Agente informativo. Usa solo il contenuto nei documenti."},
            {"role": "system", "content": "\n---\n".join(blocchi)}
        ] + hist
        r = client.chat.completions.create(model=MODEL_FT, messages=p)
        risposta = r.choices[0].message.content.strip()
        return risposta or "Non ho trovato informazioni utili sulla tua domanda."
    except Exception as e:
        print("ERRORE in agente_rag:", e)
        return "Sto riscontrando problemi tecnici. Riprova tra poco."

def agente_generico(hist):
    try:
        # Estrai il nome dall'history, MA SOLO se l'utente lo ha detto esplicitamente
        nome_utente = None
        for msg in reversed(hist):
            if "content" in msg:
                content = msg["content"].lower()
                if "mi chiamo" in content:
                    words = content.split("mi chiamo")[-1].strip().split()
                    if words:
                        nome_utente = words[0].capitalize()
                        break
                elif "chiamo" in content and "sono" not in content:
                    words = content.split("chiamo")[-1].strip().split()
                    if words:
                        nome_utente = words[0].capitalize()
                        break

        # --- LOGICA CRITICA: SE NON HO IL NOME, NON INVENTARLO ---
        if nome_utente is None:
            # Se non ho il nome, non usare "Utente" o "Giulia" — usa una risposta neutra
            system_content = (
                "Tu sei SmarTina, assistente di ITS Social. "
                "L'utente non ha ancora detto il proprio nome. "
                "Non inventare mai un nome. Rispondi in modo chiaro e amichevole, ma non attribuire un nome che non è stato dato. "
                "Se l'utente chiede 'ti ricordi come mi chiamo?', rispondi: 'Non lo so ancora, puoi dirmelo?'."
            )
        else:
            system_content = (
                f"Tu sei SmarTina, assistente di ITS Social. L'utente si chiama {nome_utente}. "
                "Rispondi in modo chiaro e amichevole. "
                "L'accesso a ITS Social è aperto a tutti gli studenti ITS ed è sufficiente un'email per registrarsi. "
                "Non servono credenziali del corso."
            )

        p = [
            {"role": "system", "content": system_content}
        ] + hist

        r = client.chat.completions.create(model=MODEL_FT, messages=p)
        risposta = r.choices[0].message.content.strip()

        # --- BLOCCA QUALSIASI NOME INVENTATO ---
        # nomi_inventati = ["giulia", "carlo", "marco", "luca", "anna", "paolo", "simone", "mario", "giuseppe"]
        # for nome in nomi_inventati:
        #     if f"ciao {nome}" in risposta.lower():
        #         risposta = risposta.replace(f"Ciao {nome.capitalize()}!", "Ciao!")
        #         risposta = risposta.replace(f"ciao {nome}!", "Ciao!")

        # --- FORZA LA RISPOSTA CORRETTA PER LA DOMANDA SUL NOME ---
        if "ti ricordi come mi chiamo" in hist[-1]["content"].lower():
            if nome_utente is None:
                return "Non lo so ancora, puoi dirmelo? 😊"
            else:
                return f"Certo, ti chiami {nome_utente}! Ti serve aiuto con qualcosa?"

        return risposta or "Non ho capito bene. Puoi ripetere?"

    except Exception as e:
        print("ERRORE in agente_generico:", e)
        return "Sto riscontrando problemi tecnici. Riprova tra poco."
    
# === MEMORIA TEMPORANEA DELLE CONVERSAZIONI (in RAM) ===

sessioni_temp = {}

def salva_messaggio_temp(user_id: str, role: str, content: str):
    """Salva messaggio temporaneamente in RAM"""
    if user_id not in sessioni_temp:
        sessioni_temp[user_id] = []
    sessioni_temp[user_id].append({"role": role, "content": content})

def carica_storia_temp(user_id: str, limite: int = 20):
    """Carica storia temporanea dalla RAM"""
    return sessioni_temp.get(user_id, [])[-limite:]

# === FUNZIONI DI GESTIONE SESSIONE SU DB TEMPORANEO ===

def salva_messaggio_db(user_id: str, role: str, content: str):
    if content is None:
        content = ""
    content = content.strip()
    if not content:
        content = "[messaggio vuoto]"

    # Rimuovi emoji (opzionale, se vuoi evitare problemi)
    content = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', '', content)

    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO sessioni_temporanee (user_id, role, content) VALUES (%s, %s, %s)",
            (user_id, role, content)
        )
        conn.commit()
    finally:
        cur.close()
        conn.close()

def carica_storia_db(user_id: str, limite: int = 20):
    """Carica la storia dal DB solo se presente"""
    conn = get_db()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT role, content FROM sessioni_temporanee "
            "WHERE user_id = %s ORDER BY created_at ASC LIMIT %s",
            (user_id, limite)
        )
        rows = cur.fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]
    finally:
        cur.close()
        conn.close()

def elimina_sessione_db(user_id: str):
    """Cancella tutti i messaggi della sessione dal DB"""
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM sessioni_temporanee WHERE user_id = %s", (user_id,))
        conn.commit()
        print(f"[ELIMINATO] Sessione {user_id} cancellata dal DB.")
    finally:
        cur.close()
        conn.close()

def chiudi_sessione(user_id: str):
    """Chiude la sessione e cancella la cronologia dal DB"""
    elimina_sessione_db(user_id)
    if user_id in sessioni_temp:  # se usi anche RAM
        del sessioni_temp[user_id]
    print(f"[FINE] Sessione {user_id} chiusa. Tutti i dati temporanei sono stati eliminati.")

# === FUNZIONE PRINCIPALE ===

def smarTina_chat(user_id: str, nuovo_messaggio: str, history_esterno=None): 
    """
    Funzione principale: chat temporanea salvata nel DB e cancellata alla chiusura.
    """
    try:
        # Carica la storia dal DB (se esiste)
        history = carica_storia_db(user_id)

        # Se è il primo messaggio → invia benvenuto
        if not history:
            benvenuto = "Ciao, sono SmarTina. In cosa ti posso essere utile?"
            salva_messaggio_db(user_id, "assistant", benvenuto)
            return benvenuto

        # Aggiungi messaggio utente
        history.append({"role": "user", "content": nuovo_messaggio})
        salva_messaggio_db(user_id, "user", nuovo_messaggio)

         # Orchestratore decide
        dec = orchestratore(history)

        # Inizializza out con un valore di fallback
        out = "Errore: risposta non riconosciuta dal sistema."

        if dec.startswith("CALL:TICKET:"):
            out = agente_ticket(history)
            if out.startswith("CALL:CONFIRMED:"):
                raw = out.replace("CALL:CONFIRMED:", "", 1).strip()
                dati = validate_and_normalize_payload(raw)
                registra_ticket(
                    dati["nome_utente"],
                    dati["cognome_utente"],
                    dati["username"],
                    dati["email"],
                    dati["tipo_ticket"]
                )
                out = "Ticket registrato correttamente nel database."

        elif dec.startswith("CALL:RAG:"):
            out = agente_rag(history)

        elif dec.startswith("CALL:GEN:"):
            out = agente_generico(history)

        # Assicurati che out sia sempre una stringa valida
        if out is None:
            out = "Mi dispiace, ho avuto un problema interno. Riprova più tardi."
        out = str(out).strip()
        if not out:
            out = "Non ho capito cosa intendi. Puoi riformulare?"

        # Salva risposta dell'assistente
        salva_messaggio_db(user_id, "assistant", out)

        return out

    except Exception as e:
        return f"Errore: {e}"