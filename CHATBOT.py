# ###########################################################
# CHATBOT SENZA IL CARICAMENTO DEL PDF DA PARTE DELL'UTENTE #

from itertools import chain
from click import prompt
import streamlit as st
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Inserisci qui la tua chiave API di OpenAI
OPENAI_API_KEY = st.secrets["superkey"]

st.set_page_config(page_title= "INFO GENERALI BOT",
                   page_icon=":credit_card:")

st.markdown(
# Gestione colori esadecimali: https://divmagic.com/it/tools/color-converter
    """
    <style>
    .stApp {
        background-color: #f5fffa;
        color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True)

st.header(":credit_card: INFO GENERALI BOT :credit_card:")

from PIL import Image
logo = Image.open("leone Generali Italia.jpg")
st.image(logo, width=800)

documento = "Risorse.pdf"

with pdfplumber.open(documento) as pdf:
    st.write(f"Pagine totali: {len(pdf.pages)} - Comincio la scansione...")
    testo = ""
    for pagina in pdf.pages:
        testo += pagina.extract_text() + "\n"
    # st.write(testo)

taglierina = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " "],
    chunk_size=1000,
    chunk_overlap=200)

frammenti = taglierina.split_text(testo)
st.write(f"Totale frammenti creati: {len(frammenti)}")
# st.write(frammenti)

# Generiamo gli embeddings
embeddings = OpenAIEmbeddings(
    # https://platform.openai.com/docs/models
    model="text-embedding-3-small",
    openai_api_key=st.secrets["OPENAI_API_KEY"])

# Salviamo gli embeddings in un vector store o vector db (es. FAISS, Pinecone, etc.)
vettori = FAISS.from_texts(frammenti, embedding=embeddings)

# Richiesta utente

# --------------------------------------------------
# Gestione prompt
# --------------------------------------------------

def invia():
    st.session_state.domanda_inviata = st.session_state.domanda_utente
    # salva il contenuto di input, cioè domanda_utente, in domanda_inviata
    st.session_state.domanda_utente = ""
    # reset dopo invio

st.text_input("Chiedi al chatbot:", key="domanda_utente", on_change=invia)
# key="domanda_utente": assegna a st.session_state ciò che scriviamo (domanda_utente)
# Ogni volta che l’utente modifica il campo e preme Invio,
# la funzione invia() viene chiamata.

domanda_utente = st.session_state.get("domanda_inviata", "")
# Recupera il valore salvato in "domanda_inviata".
# Se "domanda_inviata" non è ancora stato definito (es. al primo avvio dell'app),
# allora il valore predefinito sarà "" (secondo argomento dell'istruzione)

# --------------------------------------------------

# Generazione della risposta in una chain
# domanda -> embedding -> similarity search -> risultati all'LLM -> risposta
    
def formatta_documento(documenti):
    return "\n\n".join([documento.page_content for documento in documenti])

comparatore = vettori.as_retriever(
    # mmr = maximal marginal relevance
    search_type="mmr",
    # Ritorna i 4 frammenti più simili
    search_kwargs={"k": 4})
    
modello_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    max_tokens=1000,
    openai_api_key=st.secrets["superkey"])

prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "Sei un agente assicurativo." 
     "Usa il contesto fornito per rispondere alla domanda in modo conciso."
     "Non accedere a informazioni esterne, come Internet."
     "Se non conosci la risposta, dì semplicemente 'Non lo so'."
     "contesto:\n{context}"),
    ("human", "{question}")
    ])   

catena = (
    {"context": comparatore | formatta_documento, "question": RunnablePassthrough()}
    | prompt
    | modello_llm
    | StrOutputParser()
    )
    
if domanda_utente:
    risposta = catena.invoke(domanda_utente)
    st.write(risposta)
