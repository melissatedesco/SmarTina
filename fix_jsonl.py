#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ripara smartina_finetune_v2.jsonl:
  1. Splitta le righe con piu' oggetti JSON incollati insieme
  2. Corregge l'encoding corrotto (latin-1 letto come utf-8)
  3. Valida ogni riga e scarta quelle non riparabili
  4. Salva il file pulito
"""

import json
import re

INPUT  = "smartina_finetune_v2.jsonl"
OUTPUT = "smartina_finetune_v2.jsonl"
BACKUP = "smartina_finetune_v2_backup.jsonl"

# --- backup ---
with open(INPUT, "r", encoding="utf-8") as f:
    raw = f.read()
with open(BACKUP, "w", encoding="utf-8") as f:
    f.write(raw)
print(f"Backup salvato in: {BACKUP}")

# --- step 1: separa gli oggetti JSON incollati ---
# Due oggetti JSON sulla stessa riga appaiono come }{  →  inserisce newline
raw_fixed = re.sub(r'\}\s*\{', '}\n{', raw)

# --- step 2: correggi encoding corrotto ---
def fix_encoding(s: str) -> str:
    """
    Alcune stringhe sono state codificate male: UTF-8 salvato come latin-1.
    Il pattern tipico e': 'puÃ²' → 'può', 'Ã¨' → 'è', ecc.
    Si corregge rileggendo i byte della stringa come latin-1 e decodificando come utf-8.
    """
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return s  # se non funziona lascia invariato

# --- step 3: processa riga per riga ---
lines = [l.strip() for l in raw_fixed.splitlines() if l.strip()]

good   = []
broken = []

for i, line in enumerate(lines, 1):
    # prova prima senza correzione encoding
    try:
        obj = json.loads(line)
        good.append(json.dumps(obj, ensure_ascii=False))
        continue
    except json.JSONDecodeError:
        pass

    # prova correggendo l'encoding
    fixed_line = fix_encoding(line)
    try:
        obj = json.loads(fixed_line)
        good.append(json.dumps(obj, ensure_ascii=False))
        continue
    except json.JSONDecodeError:
        pass

    # non riparabile
    broken.append((i, line[:80]))

# --- step 4: scrivi il file pulito ---
with open(OUTPUT, "w", encoding="utf-8") as f:
    for entry in good:
        f.write(entry + "\n")

print(f"\nRisultato:")
print(f"  Esempi validi salvati : {len(good)}")
print(f"  Righe non riparabili  : {len(broken)}")
if broken:
    print("\nRighe scartate:")
    for n, preview in broken:
        print(f"  riga {n}: {preview}...")

print(f"\nFile aggiornato: {OUTPUT}")
print("Ora puoi rilanciare smarTina-gpt-4o-mini-ft.py per il fine-tuning.")
