import streamlit as st
import os
import json
import shutil
import hashlib

# Ollama
from langchain_ollama import ChatOllama, OllamaEmbeddings

# LangChain
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain_chroma import Chroma  # pip install langchain-chroma
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Agent LangGraph (fichier agent.py, dans le même dossier)
from agent import build_agent, stream_agent

# --------------------------------------------------
# Configuration
# --------------------------------------------------
st.set_page_config(page_title="Mon IA Offline (RAG)", layout="wide")
st.title("📂 Assistant Local (RAG)")

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
DATA_DIR_PATH = "data/"
PARENT_STORE_DIR = "./parent_store"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "split_parents"
REGISTRY_PATH = "./index_registry.json"  # {chemin_fichier: hash_sha256}

LOADERS = {
    ".pdf": PyPDFLoader,
    ".docx": UnstructuredWordDocumentLoader,
}


@st.cache_resource
def load_models():
    llm = ChatOllama(
        model="llama3.2:3b",
        base_url=OLLAMA_URL,
        temperature=0.4,
        num_ctx=8192,  # fenêtre de contexte élargie : évite la troncature silencieuse
    )
    # Même modèle et même num_ctx (pas de rechargement), mais sortie JSON pour le routeur et le vérificateur
    llm_json = ChatOllama(
        model="llama3.2:3b",
        base_url=OLLAMA_URL,
        temperature=0,
        num_ctx=8192,
        format="json",
    )
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_URL)
    return llm, llm_json, embeddings


llm, llm_json, embeddings = load_models()


# --------------------------------------------------
# Registre d'indexation (déduplication par hash de fichier)
# --------------------------------------------------
def load_registry():
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_registry(registry):
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def list_data_files():
    files = []
    for root, _, names in os.walk(DATA_DIR_PATH):
        for name in names:
            if os.path.splitext(name)[1].lower() in LOADERS:
                files.append(os.path.join(root, name))
    return files


# --------------------------------------------------
# Stockage : parents (disque) + enfants (Chroma)
# --------------------------------------------------
def get_docstore():
    os.makedirs(PARENT_STORE_DIR, exist_ok=True)
    if "docstore" not in st.session_state:
        st.session_state.docstore = create_kv_docstore(LocalFileStore(PARENT_STORE_DIR))
    return st.session_state.docstore


def get_vectorstore():
    if st.session_state.get("vectorstore") is None:
        st.session_state.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )
    return st.session_state.vectorstore


def get_retriever():
    return ParentDocumentRetriever(
        vectorstore=get_vectorstore(),
        docstore=get_docstore(),
        child_splitter=RecursiveCharacterTextSplitter(chunk_size=400),
        parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000),
    )


def index_new_files():
    """Indexe uniquement les fichiers nouveaux ou modifiés. Retourne (nb_indexés, nb_ignorés)."""
    registry = load_registry()
    retriever = get_retriever()
    indexed, skipped = 0, 0

    for path in list_data_files():
        h = file_hash(path)
        if registry.get(path) == h:
            skipped += 1
            continue
        loader_cls = LOADERS[os.path.splitext(path)[1].lower()]
        docs = loader_cls(path).load()
        if docs:
            retriever.add_documents(docs)
            registry[path] = h
            indexed += 1

    save_registry(registry)
    return indexed, skipped


def reset_everything():
    """Vide Chroma, les documents parents ET le registre."""
    get_vectorstore().delete_collection()
    st.session_state.vectorstore = None

    shutil.rmtree(PARENT_STORE_DIR, ignore_errors=True)
    st.session_state.pop("docstore", None)

    if os.path.exists(REGISTRY_PATH):
        os.remove(REGISTRY_PATH)


# --------------------------------------------------
# Barre latérale
# --------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    if st.button('📥 Indexer les fichiers du dossier "data"'):
        with st.spinner("Indexation en cours..."):
            if not list_data_files():
                st.warning("Aucun fichier PDF/DOCX trouvé dans le dossier /data")
            else:
                indexed, skipped = index_new_files()
                st.success(f"{indexed} fichier(s) indexé(s), {skipped} déjà à jour.")

    if st.button("🗑️ Vider la base de données"):
        try:
            reset_everything()
            st.success("Base vidée (vecteurs, parents et registre).")
            st.rerun()
        except Exception as e:
            st.error(f"Erreur lors du vidage : {e}")

    st.divider()
    st.header("💾 Exportation")

    if st.session_state.get("messages"):
        chat_export = ""
        for msg in st.session_state.messages:
            role = "Utilisateur" if msg["role"] == "user" else "Assistant"
            chat_export += f"{role}: {msg['content']}\n\n"

        st.download_button(
            label="📥 Télécharger la conversation",
            data=chat_export,
            file_name="conversation_ia.txt",
            mime="text/plain",
        )
    else:
        st.info("Lancez une discussion pour pouvoir l'exporter.")

# --------------------------------------------------
# Interface de chat
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --------------------------------------------------
# Entrée utilisateur → agent
# --------------------------------------------------
if prompt := st.chat_input("Posez votre question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not load_registry():
        st.error("Veuillez d'abord indexer les documents dans la barre latérale.")
    else:
        with st.chat_message("assistant"):
            history = st.session_state.messages[-5:-1]  # 4 derniers messages avant la question
            graph = build_agent(llm, llm_json, get_retriever())
            answer, context = "", ""
            try:
                with st.status("L'agent réfléchit...", expanded=True) as status:
                    for node, delta in stream_agent(graph, prompt, history):
                        if delta.get("log"):
                            st.write(delta["log"])
                        if delta.get("context"):
                            context = delta["context"]
                        if delta.get("answer"):
                            answer = delta["answer"]
                    status.update(label="Terminé", state="complete", expanded=False)

                if context:
                    with st.expander("📄 Documents utilisés"):
                        st.text(context[:3000])
                st.markdown(answer or "Je n'ai pas pu produire de réponse.")
            except Exception as e:
                answer = f"Erreur de l'agent : {e}"
                st.error(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})
