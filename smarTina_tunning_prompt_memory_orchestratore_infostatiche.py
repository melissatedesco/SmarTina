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

# === 2. KNOWLEDGE BASE (DATI STATICI) ===
KNOWLEDGE = {
    "its_cadmo": (
        "L'ITS CADMO ha sede a Soverato (CZ) ed è specializzato in ICT. "
        "Corsi attivi: Data Analyst & AI Specialist, Software Developer, Digital Media Designer, "
        "Digital & Energy Process Specialist, Cybersecurity Expert. Sito: https://www.itscadmo.it/"
    ),
    "calabria": (
        "ITS Academy in Calabria: ITS Cadmo, Its Efficienza Energetica, Its Pegasus, "
        "Its Tirreno, Its Pinta, Its M.A.SK., Its Iridea, Its Elaia Calabria."
    ),
    "social": (
        "ITSSocial è la piattaforma per gli studenti ITS. "
        "Funzioni: Home (post e stelle), Profilo, Tendenze. "
        "Contatti: socialitsinfo@gmail.com."
    ),
    "didattica": (
        "Il Social include classi gestite dai professori. "
        "Le classi possono essere: "
        "- CHIUSE: dedicate esclusivamente alla classe specifica del professore per materiali riservati.\n"
        "- APERTE: dove i professori caricano video didattici (es. da YouTube) su linguaggi di programmazione e altri temi tecnici."
    )
}

# === 3. ROUTER DI CONTESTO ===
def seleziona_contesto(u_input):
    u = u_input.lower()
    contesto = ""
    
    if any(k in u for k in ["cadmo", "soverato", "iscriz", "informatica", "digitale"]): 
        contesto += KNOWLEDGE["its_cadmo"] + "\n"
        
    if any(k in u for k in ["calabria", "elenco", "quali sono", "altri", "sede"]): 
        contesto += KNOWLEDGE["calabria"] + "\n"
        
    if any(k in u for k in ["social", "piattaforma", "stelle", "post", "tendenze"]): 
        contesto += KNOWLEDGE["social"] + "\n"

    # Nuova regola per le classi e i video
    if any(k in u for k in ["classe", "prof", "video", "youtube", "programmazione", "lezione"]): 
        contesto += KNOWLEDGE["didattica"] + "\n"
    
    return contesto if contesto else "Sii amichevole e rispondi come SmarTina."

# === 4. PROMPT E MEMORIA ===
storia_chat = []
memoria_utente = {"nome": ""}

prompt = ChatPromptTemplate.from_messages([
    ("system", "Sei SmarTina, l'assistente ufficiale di ITSSocial e degli ITS calabresi. "
               "IMPORTANTE: Non menzionare mai eventi, workshop o calendari, poiché non sono attualmente gestiti. "
               "Limitati a fornire supporto sulla piattaforma Social (post, stelle, profilo) e sui corsi dell'ITS CADMO.\n\n"
               "CONTESTO REALE (Usa SOLO queste info):\n{context}\n\n"
               "DATI UTENTE:\n{user_info}\n\n"
               "Sii sintetica e non fare promesse su funzionalità non presenti nel contesto."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | llm | StrOutputParser()

# === 5. LOOP PRINCIPALE ===
print("🚀 SmarTina Online! (Versione Stabile senza bug di import)")
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