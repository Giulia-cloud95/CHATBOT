#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# pip install -U streamlit PyPDF2 faiss-cpu
# pip install -U "langchain>=0.2" "langchain-openai>=0.1.7" "langchain-community>=0.2"

from itertools import chain 
from click import prompt
import streamlit as st
import pdfplumber
from langchain_text_splitters import
RecursiveCharacterTextSplitter
from langchain_openai import
OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores
import FAISS
from langchain_core.prompts import
ChatPromptTemplate
from langchain_core.runnables import
RunnablePassThrough
from langchain_core.output_parsers import
StrOutputParser

# Personalizzazioni CSS
st.set_page_config(page_title= "INFO GENERALI BOT",
                   page_icon=":credit_card:")
st.markdown(
    """
    <style>
    .stApp {
        background-color: 	#f5fffa;
        color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True)

chiave = st.secrets["superkey"]

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
#from langchain.vectorstores import FAISS
from langchain_openai import OpenAI
# Nuovi import:
#from langchain.question_answering import load_qa_chain
#from langchain_community.chatmodels import ChatOpenAI

st.header(":credit_card: INFO GENERALI BOT :credit_card:")

from PIL import Image
logo = Image.open("leone Generali Italia.jpg")
st.image(logo, width=800)
documento = "Risorse.pdf"
# st.image(logo, use_column_width=True)

with pdfplumber.open(documento) as pdf: 
  st.write(f"Pagine totali: {len(pdf.pages)} - Comincio la scansione...")
  testo = ""
  for pagina in pdf.pages:
    testo += pagina.extract_text() + "n"
    # st.write (testo)

taglierina= RecursiveCharacterTextSplitter(
      separators=["\n\n", "\n", ". ", " "],
      chunk_size=1000,
      chunk_overlap=200)

frammenti = taglierina.split_text(testo)
st.write(f"Totale frammenti creati: {len(frammenti)}")
# st.write(frammenti)


with st.sidebar:
  st.title("Carica i tuoi documenti")
  file = st.file_uploader("Carica il tuo file", type="pdf")
file = "Risorse.pdf"
  #from PyPDF2 import PdfReader
if file is not None:
  #testo_letto = PdfReader(file)
  testo = ""
  #for pagina in testo_letto.pages:
    #testo = testo + pagina.extract_text()
    
  # st.write(testo)

    # Usiamo il text splitter di Langchain
  #testo_spezzato = RecursiveCharacterTextSplitter(
    #separators="\n",
    #chunk_size=1000, # Numero di caratteri per chunk
    #chunk_overlap=150,
    #length_function=len
   #)
  #pezzi = testo_spezzato.split_text(testo)
  #st.write(pezzi)

    # Generazione embeddings
  embeddings = OpenAIEmbeddings(
      # https: //platform.openai.com/docs/models
      model="text-embedding-3-small",
      open_ap_key=st.secrets["OPENAI_API_KEY"]
    

    # Vector store - FAISS (by Facebook)
  #vector_store = FAISS.from_texts(pezzi, embeddings)
  vettori= FAISS.from_texts(frammenti,embedding=embeddings)

  # Richiesta utente

# --------------------------------------------------
# Gestione prompt
# --------------------------------------------------

  def invia():
      st.session_state.domanda_inviata = st.session_state.domanda
          # salva il contenuto di input in domanda_inviata
      st.session_state.domanda_utente = ""
          # reset dopo invio
    
  st.text_input("Chiedi al chatbot:", key="domanda_utente":, on_change=invia)
        # key="domanda_utente": assegna a st.session_state ciò che scriviamo (domanda_utente)
        # Ogni volta che l’utente modifica il campo e preme Invio,
        # la funzione invia() viene chiamata.
    
  domanda_utente = 
  st.session_state.get("domanda_inviata", "")
        # Recupera il valore salvato in "domanda_inviata".
        # Se "domanda_inviata" non è ancora stato definito (es. al primo avvio dell'app),
        # allora il valore predefinito sarà "" (secondo argomento dell'istruzione)
    
    # --------------------------------------------------

# Generazione della risposta
# domanda -> embedding -> similarity search -> risultati all'LLM -> risposta

def formatta_documento(documenti): return 
  "\n\n".join(documento.page_content for documento in documenti]) 

comparatore = vettori.as_retriever(
    # mmr = maximal marginal relevance search_type="mmr", 
    # Ritorna i 4 frammenti simili 
    
if domanda:
    st.write("Sto cercando le informazioni che mi hai richiesto...")
    #rilevanti = vector_store.similarity_search(domanda)
    
 modello_llm = ChatOpenAI(
  openai_api_key= chiave,
  temperature = 0.3,
  max_tokens = 1000,
  model_name = "gpt-4o-mini")
  openai_api_key=st.secrets["superkey"])

      # https://platform.openai.com/docs/models/compare
  #Prompt: deve avere {context} (per i documenti) e {question}
  prompt = ChatPromptTemplate.from_messages([
      ("system",
       "Sei un agente assicurativo."
       "Usa il contesto fornito per rispondere alla domanda in modo conciso."
       "Non accedere a informazioni esterne, come Internet."
       "Se non conosci la risposta, semplicemente 'Non lo so'."
       "contesto:\n{context}"),
      ("human", "{question}")
                ]) 

catena = ( 
    {"context": comparatore | 
formatta_documento, "question":
RunnablePassThrough()}
  | prompt
  | modello_llm 
  | StrOutputParser()
  )

if domanda_utente:
   risposta = 
catena.invoke(domanda_utente)
   st.write(risposta)
  

  
    # Nuova doc chain che sostituisce load_qa_chain
    #chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
# Output
      #risposta = chain.invoke ({"context": rilevanti, "question": domanda})
      #st.write(risposta["output_text"])

