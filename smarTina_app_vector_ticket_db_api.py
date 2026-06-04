#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# smarTina_app_vector_ticket_db.py

import os
import re
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import mysql.connector
import json

# === CONFIG ===
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("Manca OPENAI_API_KEY in .env")

client = OpenAI(api_key=api_key)

MODEL_MAIN = os.getenv("MODEL_MAIN", "gpt-4o-mini")
MODEL_FT = os.getenv("MODEL_FT", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

INDEX_PATH    = os.getenv("INDEX_PATH",    "rag/its_social_faiss/its_social_faiss_index.faiss")
METADATA_PATH = os.getenv("METADATA_PATH", "rag/its_social_faiss/its_social_metadata.pkl")

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

    required = ["nome_utente", "cognome_utente", "username", "email", "tipo_ticket", "descrizione_ticket"]

    missing = [k for k in required if k not in data or not str(data[k]).strip()]
    if missing:
        raise ValueError(f"Mancano i campi: {', '.join(missing)}")

    return {
        "nome_utente": data["nome_utente"].strip(),
        "cognome_utente": data["cognome_utente"].strip(),
        "username": data["username"].strip(),
        "email": data["email"].strip(),
        "tipo_ticket": data["tipo_ticket"].strip(),
        "descrizione_ticket": data["descrizione_ticket"].strip(),
    }

# === DB ===

def get_db():
    return mysql.connector.connect(**DB_CONFIG)
    
def registra_ticket(nome, cognome, username, email, tipo, descrizione):
    """Registra un nuovo ticket nel database con data/ora automatica."""
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO ticket
            (nome_utente, cognome_utente, username, email_utente, tipo_ticket, descrizione_ticket)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (nome, cognome, username, email, tipo, descrizione)
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

def cerca_blocchi_simili(q, k=5):
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
                 "Sei l'orchestratore di SmarTina. Analizza la conversazione e decidi chi risponde.\n\n"
                 "Rispondi CALL:TICKET:<testo> se:\n"
                 "- L'utente vuole fare una segnalazione, aprire un ticket, richiedere supporto, dare feedback\n"
                 "- L'utente descrive un problema tecnico o un bug ('non funziona', 'non riesco', 'errore', 'problema')\n"
                 "- La conversazione è già in corso per raccogliere dati di un ticket\n"
                 "IMPORTANTE: se nelle ultime battute l'utente ha detto 'segnalazione', 'ticket', 'problema', usa SEMPRE CALL:TICKET.\n\n"
                 "Rispondi CALL:RAG:<testo> se la richiesta riguarda:\n"
                 "- Come funziona ITSocial (spiegazioni su home, post, classi, messaggi, notifiche, profilo, tendenze)\n"
                 "- Informazioni su ITS Academy italiani (nomi, sedi, link, regioni, aree tecnologiche)\n"
                 "- Domande che iniziano con 'come si fa', 'cos'è', 'dove trovo', 'dimmi di'\n\n"
                 "Rispondi CALL:GEN:<testo> SOLO per saluti puri o chiacchiere completamente fuori tema.\n\n"
                 "Rispondi SOLO con una di queste tre forme, niente altro."
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
         "content": "Richiedi dati: nome_utente, cognome_utente, username, email, tipo_ticket, descrizione_ticket. "
                    "La descrizione_ticket è il problema o la richiesta descritta dall’utente (estraila dalla conversazione se già presente). "
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
        blocchi = cerca_blocchi_simili(ultimo, k=5)
        if not blocchi:
            return "Al momento non ho informazioni specifiche su questa richiesta."

        contesto = "\n---\n".join(blocchi)
        p = [
            {"role": "system", "content": (
                "Sei SmarTina, l'assistente ufficiale di ITSocial. "
                "Rispondi usando le informazioni nei documenti qui sotto. "
                "Puoi rispondere su: funzionalità di ITSocial (home, post, classi, messaggi, notifiche, profilo, "
                "accesso, registrazione, tendenze, like, commenti, ticket), "
                "e sugli ITS Academy italiani (cos'è un ITS, aree tecnologiche, diploma EQF 5, "
                "elenco per regione con link ufficiali). "
                "Se l'informazione è nei documenti, rispondi in modo chiaro e completo. "
                "Non inventare dati o link non presenti nel contesto.\n\n"
                "REGOLA FONDAMENTALE per le domande sugli ITS:\n"
                "- Prima risposta: elenca SOLO i nomi degli istituti (niente link, niente descrizioni). "
                "Poi chiedi esplicitamente: 'A quale di questi sei interessato?'\n"
                "- Se nella conversazione l'utente ha già indicato un istituto specifico (per nome o numero), "
                "rispondi con il link ufficiale e i dettagli di QUELL'istituto.\n"
                "- Non fornire mai link nella prima risposta. I link si danno SOLO dopo che l'utente ha scelto.\n"
                "- Non inventare nomi o link non presenti nei documenti. "
                "Se non hai dati su una scuola specifica, dillo chiaramente."
            )},
            {"role": "system", "content": f"Documenti di riferimento:\n{contesto}"}
        ] + hist
        r = client.chat.completions.create(model=MODEL_FT, messages=p)
        risposta = r.choices[0].message.content.strip()
        return risposta or "Non ho trovato informazioni utili sulla tua domanda."
    except Exception as e:
        print("ERRORE in agente_rag:", e)
        return "Sto riscontrando problemi tecnici. Riprova tra poco."

def agente_generico(hist):
    try:
        p = [
            {"role": "system", "content": (
                "Sei SmarTina, l'assistente virtuale di ITSocial. "
                "Parla in modo gentile e amichevole. "
                "Leggi tutta la conversazione: se l'utente ha già detto il suo nome, ricordatelo e usalo. "
                "Non inventare mai un nome che l'utente non ha dato. "
                "Se non conosci il nome e ti viene chiesto, rispondi 'Non me lo hai ancora detto, puoi dirmelo?'."
            )}
        ] + hist
        r = client.chat.completions.create(model=MODEL_FT, messages=p)
        return r.choices[0].message.content.strip() or "Non ho capito bene. Puoi ripetere?"
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

        # Aggiungi messaggio utente
        history.append({"role": "user", "content": nuovo_messaggio})
        salva_messaggio_db(user_id, "user", nuovo_messaggio)

         # Orchestratore decide
        dec = orchestratore(history)

        # Routing robusto — funziona anche se l'orchestratore omette i due punti finali
        if "CALL:TICKET" in dec:
            out = agente_ticket(history)
            if "CALL:CONFIRMED:" in out:
                raw = out[out.index("CALL:CONFIRMED:") + len("CALL:CONFIRMED:"):].strip()
                dati = validate_and_normalize_payload(raw)
                registra_ticket(
                    dati["nome_utente"],
                    dati["cognome_utente"],
                    dati["username"],
                    dati["email"],
                    dati["tipo_ticket"],
                    dati["descrizione_ticket"]
                )
                out = "Ticket registrato correttamente."

        elif "CALL:RAG" in dec:
            out = agente_rag(history)

        else:
            # CALL:GEN o qualsiasi risposta non riconosciuta → agente generico
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