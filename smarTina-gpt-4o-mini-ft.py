#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script di fine-tuning per SmarTina.

Passi:
  1. Carica smartina_finetune_v2.jsonl su OpenAI
  2. Avvia il job di fine-tuning
  3. Aspetta il completamento
  4. Stampa il nome del modello da usare nei file chatbot
"""

import time
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("❌ Manca la chiave API nel file .env")

client = OpenAI(api_key=api_key)

dataset_path = "smartina_finetune_v2.jsonl"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"❌ File dataset non trovato: {dataset_path}")

# --- Passo 1: carica il file ---
print("📤 Caricamento dataset su OpenAI...")
training_file = client.files.create(
    file=open(dataset_path, "rb"),
    purpose="fine-tune"
)
print(f"   File ID: {training_file.id}")

# --- Passo 2: avvia il job ---
print("🚀 Avvio fine-tuning...")
job = client.fine_tuning.jobs.create(
    training_file=training_file.id,
    model="gpt-4o-mini-2024-07-18",
    suffix="smartina"
)
print(f"   Job ID: {job.id}")
print("⏳ Fine-tuning in corso (può richiedere 10-30 minuti)...\n")

# --- Passo 3: aspetta il completamento ---
while True:
    job = client.fine_tuning.jobs.retrieve(job.id)
    status = job.status

    if status == "succeeded":
        model_name = job.fine_tuned_model
        print(f"\n✅ Fine-tuning completato!")
        print(f"\n🤖 Nuovo modello: {model_name}")
        print("\n📋 Aggiorna MODEL_FT in questi file:")
        print(f'   smarTina_app_vector.py                                → MODEL_FT = "{model_name}"')
        print(f'   smarTina_app_vector_ticket.py                         → MODEL_FT = "{model_name}"')
        print(f'   smarTina_tunning_prompt_memory_orchestratore_infostatiche.py → MODEL_FT = "{model_name}"')
        print(f'   smarTina_app_vector_ticket_db_api.py                  → MODEL_FT = "{model_name}"')
        break

    elif status in ("failed", "cancelled"):
        print(f"\n❌ Fine-tuning {status}.")
        if job.error:
            print(f"   Errore: {job.error}")
        break

    else:
        print(f"   Stato: {status} — riprovo tra 30 secondi...")
        time.sleep(30)
