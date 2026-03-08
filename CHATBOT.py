import streamlit as st
import pdfplumber
from PIL import Image

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ==================================================
# CONFIGURAZIONE PAGINA
# ==================================================

st.set_page_config(
    page_title="INFO GENERALI BOT",
    page_icon=":credit_card:"
)

st.markdown("""
<style>
.stApp {
    background-color: #f5fffa;
    color: #000000;
}
</style>
""", unsafe_allow_html=True)

st.header("Assistenza online")


# ==================================================
# PERCORSI FILE
# ==================================================

DOCUMENTO = "Risorse.pdf"
LOGO = "Leone Generali.png"


# ==================================================
# FUNZIONI DI SUPPORTO
# ==================================================

def formatta_documenti(documenti):
    """Converte i documenti recuperati in una stringa unica."""
    return "\n\n".join(doc.page_content for doc in documenti)


# ==================================================
# LETTURA PDF
# ==================================================

@st.cache_data(show_spinner=False)
def estrai_testo_pdf(percorso_pdf: str) -> str:
    """
    Estrae il testo da tutte le pagine del PDF.
    Ignora le pagine che non restituiscono testo.
    """
    testo_completo = ""

    with pdfplumber.open(percorso_pdf) as pdf:
        for pagina in pdf.pages:
            testo_pagina = pagina.extract_text()
            if testo_pagina:
                testo_completo += testo_pagina + "\n"

    return testo_completo.strip()


# ==================================================
# CREAZIONE CHUNKS
# ==================================================

@st.cache_data(show_spinner=False)
def crea_frammenti(testo: str):
    """
    Suddivide il testo in frammenti adatti alla ricerca semantica.
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(testo)


# ==================================================
# VECTOR STORE
# ==================================================

@st.cache_resource(show_spinner=False)
def crea_vectorstore(frammenti):
    """
    Crea l'indice vettoriale FAISS a partire dai frammenti.
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=st.secrets["superkey"]
    )

    return FAISS.from_texts(frammenti, embedding=embeddings)


# ==================================================
# MODELLO LLM
# ==================================================

@st.cache_resource(show_spinner=False)
def crea_llm():
    """
    Crea il modello di chat.
    """
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=1000,
        openai_api_key=st.secrets["superkey"]
    )


# ==================================================
# CATENA RAG
# NOTA: niente cache qui, per evitare errori di hashing
# ==================================================

def crea_catena(vettori):
    """
    Costruisce la chain RAG usando il vector store già pronto.
    """
    retriever = vettori.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4}
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Sei un agente assicurativo. "
            "Rispondi usando solo il contesto fornito. "
            "Non usare informazioni esterne. "
            "Se la risposta non è presente nel contesto, scrivi semplicemente: 'Non ho trovato una risposta precisa. Per garantirti informazioni sicure, ti suggerisco di consultare le nostre FAQ o di contattare un esperto'. "
            "Rispondi in italiano in modo chiaro e sintetico. "
            "Contesto:\n{context}"
        ),
        ("human", "{question}")
    ])

    catena = (
        {
            "context": retriever | formatta_documenti,
            "question": RunnablePassthrough()
        }
        | prompt
        | crea_llm()
        | StrOutputParser()
    )

    return catena


# ==================================================
# STATO SESSIONE
# ==================================================

if "risposta" not in st.session_state:
    st.session_state.risposta = ""

if "errore_setup" not in st.session_state:
    st.session_state.errore_setup = ""


# ==================================================
# LOGO
# ==================================================

try:
    logo = Image.open(LOGO)
    st.image(logo, width=800)
except Exception:
    pass


# ==================================================
# PREPARAZIONE PIPELINE
# ==================================================

catena = None

with st.spinner("Sto preparando il chatbot..."):
    try:
        testo = estrai_testo_pdf(DOCUMENTO)

        if not testo:
            raise ValueError("Il PDF non contiene testo leggibile.")

        frammenti = crea_frammenti(testo)

        if not frammenti:
            raise ValueError("Non è stato possibile creare frammenti dal testo del PDF.")

        vettori = crea_vectorstore(frammenti)
        catena = crea_catena(vettori)

        st.session_state.errore_setup = ""

    except Exception as e:
        st.session_state.errore_setup = f"Errore nella preparazione del chatbot: {e}"


# ==================================================
# INTERFACCIA UTENTE
# ==================================================

if st.session_state.errore_setup:
    st.error(st.session_state.errore_setup)

else:
    domanda = st.text_input("Scrivi qui la tua domanda:")

    if st.button("Invia", type="primary"):
        if not domanda.strip():
            st.warning("Inserisci una domanda prima di inviare.")
        else:
            with st.spinner("Sto cercando la risposta..."):
                try:
                    risposta = catena.invoke(domanda.strip())
                    st.session_state.risposta = risposta
                except Exception as e:
                    st.session_state.risposta = ""
                    st.error(f"Si è verificato un errore durante la risposta: {e}")

    if st.session_state.risposta:
        st.write("**Risposta:**")
        st.write(st.session_state.risposta)
