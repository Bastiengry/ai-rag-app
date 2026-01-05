import streamlit as st
import os
import shutil
import stat
import hashlib

# Ollama
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# LangChain (0.3.x – structure optimisée)
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# --------------------------------------------------
# Configuration Streamlit
# --------------------------------------------------
st.set_page_config(page_title="Mon IA Offline (RAG)", layout="wide")
st.title("📂 Assistant Local (RAG)")

# --------------------------------------------------
# Configuration Ollama
# --------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
DATA_DIR_PATH = "data/"
PARENT_STORE_DIR = "./parent_store"

# --------------------------------------------------
# Prompt RAG (IMPORTANT : {multi_doc_context}, {prompt} (et {history}))
# --------------------------------------------------
template = """
Tu es un assistant expert en documentation d'entreprise. 
Réponds UNIQUEMENT en utilisant les documents fournis. 
Si l'information n'est pas présente, dis que tu ne sais pas.
Cite toujours la source du document (ex: [Source: fichier.pdf]).

DOCUMENTS DE RÉFÉRENCE :
{multi_doc_context}
"""


QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Création du stockage local pour les parents (les enfants sont stockés en base Chroma, 
# mais les parents sont stockés dans des fichiers sur le disque)
if not os.path.exists(PARENT_STORE_DIR):
        os.makedirs(PARENT_STORE_DIR)

@st.cache_resource
def load_models():
    llm = ChatOllama(
        model="llama3.2:3b",
        base_url=OLLAMA_URL,
        temperature=0.2
    )
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=OLLAMA_URL
    )
    return llm, embeddings

llm, embeddings = load_models()

# Cette fonction lit le contenu des fichiers du répertoire "data" pour différents formats (PDF, docx...)
def read_data_dir_files():    
    # Chargement des PDF
    pdf_loader = DirectoryLoader(DATA_DIR_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    
    # Chargement des DOCX
    docx_loader = DirectoryLoader(DATA_DIR_PATH, glob="**/*.docx", loader_cls=UnstructuredWordDocumentLoader)
    
    # On combine les documents des deux sources
    pdf_docs = pdf_loader.load()
    docx_docs = docx_loader.load()
    
    docs = pdf_docs + docx_docs
    
    
    return docs

# Cette fonction va formater chaque morceau de texte pour inclure sa source
def format_docs_with_sources(docs):
    formatted = []
    for doc in docs:
        source_name = os.path.basename(doc.metadata.get('source', 'Inconnu'))
        content = f"[Source: {source_name}]\n{doc.page_content}"
        formatted.append(content)
    return "\n\n---\n\n".join(formatted)

def prepare_docs_with_metadata(docs):
    new_docs = []
    for doc in docs:
        source_name = os.path.basename(doc.metadata.get("source", "Inconnu"))

        doc_id = hashlib.md5(doc.metadata["source"].encode()).hexdigest()

        new_doc = Document(
            page_content=doc.page_content,
            metadata={
                "source": source_name,
                "doc_id": doc_id
            }
        )
        new_docs.append(new_doc)
    return new_docs

# Nous avons besoin d'un magasin pour stocker les documents "Parents" 
# car Chroma ne stockera que les "Enfants" (vecteurs)
if "docstore" not in st.session_state:
    fs = LocalFileStore(PARENT_STORE_DIR)
    st.session_state.docstore = create_kv_docstore(fs)

def get_retriever():
    # 1. Dossier pour stocker les documents Parents sur le disque
    
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

    # 3. Initialisation du retriever
    retriever = ParentDocumentRetriever(
        vectorstore=st.session_state.vectorstore,
        docstore=st.session_state.docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    return retriever

# --------------------------------------------------
# Barre latérale : indexation
# --------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    if st.button("📥 Indexer les fichiers du dossier \"data\""):
        with st.spinner("Indexation en cours..."):
            docs = read_data_dir_files()

            if not docs:
                st.warning("Aucun fichier trouvé dans le dossier /data")
            else:
                docs = prepare_docs_with_metadata(docs)
                
                # On initialise le vectorstore s'il n'existe pas
                if "vectorstore" not in st.session_state:
                    st.session_state.vectorstore = Chroma(
                        collection_name="split_parents",
                        embedding_function=embeddings,
                        persist_directory="./chroma_db"
                    )
                
                retriever = get_retriever()
                # Cette commande découpe en parents/enfants et indexe tout automatiquement
                ids = [doc.metadata["doc_id"] for doc in docs]
                retriever.add_documents(docs, ids=ids)
                st.session_state.vectorstore.persist()
                st.success("Indexation Parent-Enfant terminée.")

    if st.button("🗑️ Vider la base de données"):
        # On réinitialise la référence dans Streamlit
        if "vectorstore" in st.session_state:
            st.session_state.vectorstore = None

        # Supprimer Chroma ET le Parent Store
        try:
            for path in ["./chroma_db", "./parent_store"]:
                if os.path.exists(path):
                    # Sous Windows, parfois il faut forcer la suppression
                    def remove_readonly(func, p, _):
                        os.chmod(p, stat.S_IWUSR)
                        func(p)
                    shutil.rmtree(path, onerror=remove_readonly)
            st.success("Base de données vidée avec succès. Veuillez ré-indexer.")
            st.rerun()
        except Exception as e:
            st.error(f"Erreur suppression {path}: {e}")
        
    st.divider() # Petite ligne de séparation visuelle
    st.header("💾 Exportation")
    
    if "messages" in st.session_state and st.session_state.messages:
        # On prépare le contenu du fichier texte
        chat_export = ""
        for msg in st.session_state.messages:
            role = "Utilisateur" if msg["role"] == "user" else "Assistant"
            chat_export += f"{role}: {msg['content']}\n\n"
        
        st.download_button(
            label="📥 Télécharger la conversation",
            data=chat_export,
            file_name="conversation_ia.txt",
            mime="text/plain"
        )
    else:
        st.info("Lancez une discussion pour pouvoir l'exporter.")


# --------------------------------------------------
# Chargement du vectorstore existant (si présent)
# --------------------------------------------------
if "vectorstore" not in st.session_state and os.path.exists("./chroma_db"):
    st.session_state.vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

# --------------------------------------------------
# Interface de chat
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --------------------------------------------------
# Entrée utilisateur
# --------------------------------------------------
if prompt := st.chat_input("Posez votre question..."):    
    try:
        # Affichage et stockage du message utilisateur
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        if "vectorstore" not in st.session_state:
            st.error("Veuillez d'abord indexer les documents dans la barre latérale.")
        else:
            with st.chat_message("assistant"):
                # On récupère le retriever
                retriever = get_retriever()

                # PLUS BESOIN de similarity_search_with_score ni de THRESHOLD
                # On demande au retriever de trouver les documents les plus pertinents
                # Il va chercher les petits enfants et nous rendre les gros parents.
                context_docs = retriever.invoke(prompt)
                if not context_docs:
                    st.error("Aucune information trouvée dans les documents.")
                else:
                    # Construction du contexte avec les documents parents complets
                    multi_doc_context = ""
                    for i, doc in enumerate(context_docs):
                        source = doc.metadata.get("source", "Inconnu")
                        multi_doc_context += f"Document {i+1} (Source: {source}):\n{doc.page_content}\n\n"
        
                    with st.expander("📄 Documents utilisés"):
                        st.text(multi_doc_context[:3000])

                    # On commence par l'instruction système avec les documents
                    messages = [
                        SystemMessage(content=QA_CHAIN_PROMPT.format(multi_doc_context=multi_doc_context))
                    ]

                    # Ajout de l'historique récent (optionnel mais recommandé pour l'affinement)
                    # On transforme les dictionnaires Streamlit en objets Messages LangChain
                    for msg in st.session_state.messages[-5:-1]: # Les 4 derniers messages
                        if msg["role"] == "user":
                            messages.append(HumanMessage(content=msg["content"]))
                        else:
                            messages.append(AIMessage(content=msg["content"]))

                    # Ajout de la question actuelle
                    messages.append(HumanMessage(content=prompt))
                    
                    answer = ""
                    with st.spinner("L'IA analyse les documents..."):
                        try:
                            placeholder = st.empty()
                            full_answer = ""
                            
                            # Utilisation du stream pour un affichage fluide
                            for chunk in llm.stream(messages):
                                full_answer += chunk.content # .content est nécessaire avec ChatOllama
                                placeholder.markdown(full_answer + "▌")
                            
                            placeholder.markdown(full_answer)
                            answer = full_answer
                        except Exception as e:
                            st.error(f"Erreur Ollama : {e}")
                            answer = "Erreur lors de la génération."

                    # On enregistre la réponse finale dans l'historique
                    st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        st.error(f"Erreur: {e}")