import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# === 1. CONFIGURAZIONE ===

load_dotenv(override=True)

# Usiamo il nuovo nome della variabile per evitare conflitti di sistema
api_key = os.getenv("SMARTINA_KEY", "").strip()

MODEL_FT = "ft:gpt-4o-mini-2024-07-18:its-cadmo:smartina:CcpM9wrx"

if not api_key.startswith("sk-"):
    print("❌ Errore: Non riesco a leggere SMARTINA_KEY dal file .env")
    exit()

llm = ChatOpenAI(
    model=MODEL_FT, 
    temperature=0.7, 
    openai_api_key=api_key
)

# Usiamo il tuo modello Fine-Tuned
MODEL_FT = "ft:gpt-4o-mini-2024-07-18:its-cadmo:smartina:CcpM9wrx"
llm = ChatOpenAI(model=MODEL_FT, temperature=0.7, openai_api_key=api_key)

# === 2. KNOWLEDGE BASE  ===
KNOWLEDGE = {
    "calabria": "Calabria: ITS CADMO (CZ), Efficienza Energetica (RC), Pegasus (RC),"
        "Tirreno (CS), Pinta (KR), M.A.SK. (RC), Iridea (CS), Elaia (VV).",
        
    "campania": "Campania: BACT (NA), TEC MOS (CE), Mobilità Sostenibile (NA), Moda Campania (NA), MA.ME. (NA), Energy-Lab (BN), "
        "Antonio Bruno (AV), Ermete (AV), SCI.TEC.VITA (NA), Newtech SI (SA), "
        "TE.LA. (SA), Casa Campania (NA), Hitech & Communication (NA), "
        "ICT Campus (BN), Ma.De. Academy (NA).",

    "emilia_romagna": "Emilia-Romagna: Logistica e Mobilità (PC), "
        "Agroalimentare (PR), Meccanica Meccatronica (BO), "
        "Territorio Energia (FE),Tecnologie Industrie Creative (FC), "
        "Turismo e Benessere (RN), Nuove Tecnologie della Vita (MO).",

    "friuli": "Friuli Venezia Giulia: Meccanica e Aeronautica (UD), "
        "Alto Adriatico (PN), Alessandro Volta (TS), Accademia Nautica Adriatico (TS).",

    "lazio": "Lazio: Caboto (LT), Agroalimentare (VT), Rossellini (RM), "
        "Servizi Imprese (VT), Bio Campus (LT), Tecnologie Vita (RM), Turismo (RM), "
        "Meccatronico (FR), Lazio Digital (RM), Agnesi (RM), ICT Academy (RM), "
        "Sistema Moda (RM), Agroalimentare Rieti (RI), Eco-Stem (RM), ITSEL (RM), "
        "Logistica 4.0 (RI).",

    "liguria": "Liguria: Efficienza Energetica (SV), ICT Liguria (GE), "
        "Accademia Marina Mercantile (GE), Meccanico/Navalmeccanico (SP), "
        "Agroalimentare (IM), Turismo Liguria (GE).",

    "lombardia": "Lombardia: Rizzoli (MI), JobsAcademy (BG), Minoprio (CO), Jobs Factory (PV),"
        " Tecnologie Vita (BG), Machina Lonati (BS), Trasporti/Logistica (VA), "
        "Cantieri dell’Arte (MI), Made in Italy (CR), Energia Ambiente (MB), "
        "Turismo (CO), Meccatronica (MI), Sistema Casa (MB), "
        "Agroalimentare (LO/MN/SO), InnovaProfessioni (MI), TTF (MI), Symposium (BS), "
        "I-CREA (BG).",

    "marche_molise": "Marche: Recanati (MC), Moda (FM), Energia (AN), "
        "Turismo (PU). Molise: D.E.Mo.S. (CB).",

    "piemonte": "Piemonte: ICT (TO), Mobilità Aerospazio (TO), Sistema Moda "
        "(BI), Agroalimentare (CN), Biotecnologie (TO), "
        "Sistemi Energetici (TO), Turismo (TO).",
        
    "puglia": "Puglia: Aerospazio (BR), Cuccovillo (BA), Agroalimentare (BA), "
        "Apulia Digital Maker (FG), Turismo (LE), "
        "Infomobilità (TA), MI.TI (TA).",

    "sardegna": "Sardegna: Efficienza Energetica (NU), Mo.So.S. (CA), "
        "Agroalimentare (SS), Turismo (SS), Novitas 4.0 (NU).",

    "sicilia": "Sicilia: Enna (EN), Steve Jobs (CT), Albatros (ME), "
        "Archimede (SR), Trasporti (CT), Alessandro Volta (PA), Sicani (AG), "
        "Emporium del Golfo (TP), Aerospazio (RG), Madonie (PA).",

    "toscana": "Toscana: Energia Ambiente (SI), Prime (FI), M.I.T.A. (FI), "
    "ISYL (LU), E.A.T. (GR), VITA (SI), TAB (FI), Prodigi (FI), A.T.E. (LI).",

    "umbria_veneto": "Umbria: Made in Italy (PG). Veneto: Turismo (VE), "
    "Mobilità (VR), Meccatronico (VI), RED (PD), Moda (PD), "
    "Agroalimentare (TV), Marco Polo (VE), Digital Academy (PD).",

    "social_didattica": "ITSSocial: Home (post/stelle), Profilo, Tendenze. Didattica: Classi CHIUSE (riservate) e APERTE (video didattici YouTube su programmazione)."
}
def seleziona_contesto(u_input):
    u = u_input.lower()
    contesto = ""
    # Mappa delle parole chiave per ogni regione
    mappa_regioni = {
        "campania": "campania", "lazio": "lazio", "lombardia": "lombardia", 
        "sicilia": "sicilia", "piemonte": "piemonte", "puglia": "puglia", 
        "toscana": "toscana", "sardegna": "sardegna", "liguria": "liguria",
        "emilia": "emilia_romagna", "romagna": "emilia_romagna",
        "friuli": "friuli", "veneto": "umbria_veneto", "umbria": "umbria_veneto",
        "marche": "marche_molise", "molise": "marche_molise", "calabria": "calabria",
    }
    
    for chiave, regione in mappa_regioni.items():
        if chiave in u:
            contesto += KNOWLEDGE[regione] + "\n"
    
    if any(k in u for k in ["social", "stelle", "post", "video", "classe"]):
        contesto += KNOWLEDGE["social_didattica"] + "\n"
    
    return contesto if contesto else "Informazioni generali sugli ITS d'Italia e ITSSocial."
# === 4. PROMPT E MEMORIA ===
storia_chat = []
memoria_utente = {"nome": ""}

prompt = ChatPromptTemplate.from_messages([
    ("system", "Sei SmarTina, l'assistente ufficiale di ITSSocial. "
               "REGOLE: \n1. Non parlare MAI di eventi, workshop o calendari.\n"
               "2. Usa esclusivamente i dati del contesto per elencare gli ITS.\n"
               "3. Sii professionale, amichevole e sintetica.\n\n"
               "CONTESTO REALE:\n{context}\n\n"
               "DATI UTENTE:\n{user_info}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])
chain = prompt | llm | StrOutputParser()

# === 5. LOOP PRINCIPALE ===
print("🚀 SmarTina l'assistente dell'ITSOCIAL")
print("Scrivi 'exit' per uscire o 'dimentica tutto' per resettare.\n")

while True:
    u_input = input("👤 Tu: ").strip()
    if not u_input: continue
    if u_input.lower() in ["exit", "quit"]: break
    
    if u_input.lower() == "dimentica tutto":
        storia_chat = []
        memoria_utente["nome"] = ""
        print("🧽 Memoria pulita!\n")
        continue

    # Gestione Nome
    if u_input.lower().startswith(("mi chiamo ", "il mio nome è ")):
        nome = u_input.split()[-1].strip().capitalize()
        memoria_utente["nome"] = nome
        print(f"💬 SmarTina: Piacere {nome}! Me lo sono segnato. 😊\n")
        continue

    # Recupero contesto
    current_context = seleziona_contesto(u_input)
    
    try:
        risposta = chain.invoke({
            "input": u_input,
            "context": current_context,
            "user_info": f"L'utente si chiama {memoria_utente['nome']}" if memoria_utente["nome"] else "Nome sconosciuto.",
            "history": storia_chat[-6:] 
        })
        
        # Aggiornamento storia
        storia_chat.append(HumanMessage(content=u_input))
        storia_chat.append(AIMessage(content=risposta))
        
        print(f"💬 SmarTina: {risposta}\n")
    except Exception as e:
        print(f"❌ Errore: {e}")