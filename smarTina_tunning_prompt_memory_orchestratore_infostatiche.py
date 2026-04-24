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
        "ITS Academy in Calabria: "
        "- ITS Cadmo (Soverato): ICT\n"
        "- Its Efficienza Energetica (Reggio Calabria): Energia\n"
        "- Its Pegasus (Polistena): Mobilità Sostenibile\n"
        "- Its Tirreno (Fuscaldo): Chimica e Nuove Tecnologie della Vita\n"
        "- Its Pinta (Cotronei): Agroalimentare\n"
        "- Its M.A.SK. (San Ferdinando): Servizi alle Imprese\n"
        "- Its Iridea (Cosenza): Agroalimentare\n"
        "- Its Elaia Calabria (Vibo Valentia): Turismo"
    ),
    "social": (
        "ITSSocial è la piattaforma per gli studenti ITS. "
        "Funzioni: Home (post e stelle), Profilo, Tendenze. "
        "Contatti: socialitsinfo@gmail.com. C'è anche una sezione video didattici."
    )
}

# === 3. ROUTER DI CONTESTO (Sostituisce i Tools) ===
def seleziona_contesto(u_input):
    u = u_input.lower()
    contesto = ""
    # Se l'utente chiede del Cadmo o di informatica
    if any(k in u for k in ["cadmo", "soverato", "iscriz", "informatica", "digitale"]): 
        contesto += KNOWLEDGE["its_cadmo"] + "\n"
    # Se chiede degli altri ITS o della Calabria in generale
    if any(k in u for k in ["calabria", "elenco", "quali sono", "altri", "sede"]): 
        contesto += KNOWLEDGE["calabria"] + "\n"
    # Se chiede del social
    if any(k in u for k in ["social", "piattaforma", "stelle", "post", "video"]): 
        contesto += KNOWLEDGE["social"] + "\n"
    
    return contesto if contesto else "Sii amichevole e rispondi come SmarTina."

# === 4. PROMPT E MEMORIA ===
storia_chat = []
memoria_utente = {"nome": ""}

prompt = ChatPromptTemplate.from_messages([
    ("system", "Sei SmarTina, l'assistente ufficiale di ITSSocial e degli ITS calabresi. "
               "Dai sempre molta importanza all'ITS CADMO. Se l'utente chiede info su altri ITS, "
               "forniscile ma suggerisci il Cadmo come eccellenza digitale.\n\n"
               "CONTESTO DISPONIBILE:\n{context}\n\n"
               "DATI UTENTE:\n{user_info}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# Chain semplificata (No AgentExecutor = No Errori)
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