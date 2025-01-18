import os
import traceback
import logging
import psutil
import pandas as pd
from PyPDF2 import PdfReader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
import streamlit as st
import pdfplumber
import time
import sys

#SQLITE
# Importar sqlite3 desde pysqlite3 si está disponible
try:
    from pysqlite3 import dbapi2 as sqlite3
    sys.modules["sqlite3"] = sqlite3
except ImportError:
    import sqlite3  # Fallback al sqlite3 predeterminado
    
# Configuración de logging
logging.basicConfig(
    filename="debug.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_and_display_error(error_message):
    logging.error(error_message)
    st.error(error_message)

def check_system_resources():
    # Verificar si hay al menos 500 MB de memoria disponible
    if psutil.virtual_memory().available < 500 * 1024 * 1024:
        log_and_display_error("Memoria insuficiente para procesar la solicitud. Cierre otras aplicaciones e intente nuevamente.")
        return False
    return True

# Configuración de la API de OpenAI
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Configuración de la página de Streamlit
st.set_page_config(page_title="IA Chat", layout="wide")

# Mostrar el logotipo en la barra lateral
logo_path = "PCM-PCM.jpg"
with st.sidebar:
    if os.path.exists(logo_path):
        st.image(logo_path, width=400)
    else:
        st.warning("Logotipo no encontrado en la ruta especificada.")

# Mostrar el título
st.title("CHAT IMARPE")

# Inicializar el estado si no existe
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "files_processed" not in st.session_state:
    st.session_state.files_processed = False

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# Cargar la base de datos vectorial al inicio si existe
try:
    if st.session_state.vectorstore is None:
        from chromadb import Client
        client = Client()
        st.session_state.vectorstore = Chroma(
            client=client,
            persist_directory="./vectordb",
            embedding_function=OpenAIEmbeddings()
        )
        st.success("Base de datos cargada exitosamente.")
except Exception as e:
    log_and_display_error(f"Error al cargar la base de datos: {e}\n{traceback.format_exc()}")

# Procesar nuevos archivos si se suben
uploaded_files = st.sidebar.file_uploader("Sube tus documentos aquí:", type=["pdf", "xlsx", "xls"], accept_multiple_files=True)
if uploaded_files and not st.session_state.files_processed:
    all_docs = []
    max_file_size_mb = 20  # Límite de 20 MB
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.size > max_file_size_mb * 1024 * 1024:
                log_and_display_error(f"El archivo {uploaded_file.name} excede el tamaño máximo permitido de {max_file_size_mb} MB.")
                continue

            if uploaded_file.name.endswith(".pdf"):
                try:
                    with pdfplumber.open(uploaded_file) as pdf:
                        text = "".join(page.extract_text() or "" for page in pdf.pages)
                    document = Document(page_content=text, metadata={"source": uploaded_file.name})
                    all_docs.append(document)
                except Exception as e:
                    log_and_display_error(f"Error al procesar PDF {uploaded_file.name}: {e}")
            elif uploaded_file.name.endswith((".xlsx", ".xls")):
                try:
                    chunk_size = 500  # Número de filas por chunk
                    total_rows = pd.read_excel(uploaded_file).shape[0]  # Calcular número total de filas

                    for start_row in range(0, total_rows, chunk_size):
                        header = 0 if start_row == 0 else None  # Incluir encabezados solo en la primera iteración
                        chunk = pd.read_excel(uploaded_file, skiprows=start_row, nrows=chunk_size, header=header)
                        text = chunk.to_string(index=False, header=(start_row == 0))
                        document = Document(page_content=text, metadata={"source": uploaded_file.name})
                        all_docs.append(document)
                        st.write(f"Procesando chunk desde fila {start_row}")
                except Exception as e:
                    log_and_display_error(f"Error al procesar Excel {uploaded_file.name}: {e}")
        except Exception as e:
            log_and_display_error(f"Error al procesar el archivo {uploaded_file.name}: {e}\n{traceback.format_exc()}")

    if all_docs:
        st.write(f"Documentos totales procesados: {len(all_docs)}")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = text_splitter.split_documents(all_docs)
        st.write(f"Fragmentos creados: {len(splits)}")
        try:
            if st.session_state.vectorstore:
                st.session_state.vectorstore.add_documents(splits)
                st.session_state.vectorstore.persist()  # Método correcto para persistir los datos
                st.success("Nuevos documentos procesados y añadidos a la base de datos.")
                st.session_state.files_processed = True
            else:
                log_and_display_error("Vectorstore no está inicializado.")
        except Exception as e:
            log_and_display_error(f"Error al actualizar la base de datos: {e}\n{traceback.format_exc()}")
    else:
        log_and_display_error("No se procesaron documentos. Verifique el formato y contenido de los archivos cargados.")

# Mostrar historial de chat y entrada de texto en un contenedor
with st.container():
    st.subheader("Chat con tus documentos")

    # Reservar espacio para el historial de chat
    chat_placeholder = st.empty()

    # Rellenar el espacio reservado con el historial de chat
    with chat_placeholder.container():
        for i, chat in enumerate(st.session_state.chat_history):
            role, msg = chat
            # Generar una clave única usando el índice y un timestamp
            key = f"{role}_{i}_{int(time.time() * 1000)}"
            if role == "human":
                message(msg, is_user=True, key=key)
            else:
                message(msg, key=key)

    # Reservar un espacio para el formulario, siempre ubicado al final
    form_placeholder = st.empty()

    # Usar un formulario para la entrada de texto
    with form_placeholder.form("consulta_form"):
        query = st.text_input("Coloca tu pregunta en esta caja:")
        submit_button = st.form_submit_button("Enviar")

# Procesar la consulta solo al enviar el formulario
if submit_button and query:
    if query != st.session_state.last_query:
        st.session_state.last_query = query
        if st.session_state.vectorstore:
            if check_system_resources():
                try:
                    retriever = st.session_state.vectorstore.as_retriever()
                    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
                    rag_chain = ConversationalRetrievalChain.from_llm(llm, retriever)

                    # Generar respuesta a la consulta
                    response = rag_chain.invoke({"question": query, "chat_history": st.session_state.chat_history})

                    # Actualizar historial de chat
                    st.session_state.chat_history.append(("human", query))
                    st.session_state.chat_history.append(("assistant", response["answer"]))

                    # Actualizar el historial en la interfaz
                    with chat_placeholder.container():
                        for i, chat in enumerate(st.session_state.chat_history):
                            role, msg = chat
                            # Generar una clave única usando el índice y un timestamp
                            key = f"{role}_{i}_{int(time.time() * 1000)}"
                            if role == "human":
                                message(msg, is_user=True, key=key)
                            else:
                                message(msg, key=key)

                except Exception as e:
                    log_and_display_error(f"Error al realizar la consulta: {e}\n{traceback.format_exc()}")
        else:
            log_and_display_error("La base de datos no está cargada. Por favor, procesa nuevos archivos o verifica la carga de la base de datos persistente.")
    else:
        st.warning("La pregunta ya fue enviada.")

