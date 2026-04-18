#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script di addestramento per SmarTina - Step 2 (Fine-tuning)
"""

from openai import OpenAI
from dotenv import load_dotenv
import os

# Carica chiave API
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise SystemExit("❌ Manca la chiave API nel file .env")

client = OpenAI(api_key=api_key)

# Percorso dataset
dataset_path = "smarTina_dataset.jsonl"

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"❌ File dataset non trovato: {dataset_path}")

print("🚀 Avvio fine-tuning di SmarTina...")

# Carica il file su OpenAI
training_file = client.files.create(
    file=open(dataset_path, "rb"),
    purpose="fine-tune"
)

# Avvia il job di fine-tuning
job = client.fine_tuning.jobs.create(
    training_file=training_file.id,
    model="gpt-4o-mini-2024-07-18",  # modello snapshot valido per fine-tuning
    suffix="smartina"
)


print("✅ Job di fine-tuning avviato!")
print(f"🆔 Job ID: {job.id}")
print("⏳ Controlla lo stato con:")
print("   openai api fine_tunes.follow -i <job_id>")
print("   oppure gestiscilo dalla dashboard OpenAI.")