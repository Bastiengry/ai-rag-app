import streamlit as st
import os

# Ollama
from langchain_ollama import OllamaLLM, OllamaEmbeddings

# LangChain (0.3.x – structure optimisée)
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

# --------------------------------------------------
# Configuration Streamlit
# --------------------------------------------------
st.set_page_config(page_title="Mon IA Offline (RAG)", layout="wide")
st.title("📂 Assistant Local (RAG)")

# --------------------------------------------------
# Configuration Ollama
# --------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

@st.cache_resource
def load_models():
    llm = OllamaLLM(
        model="llama3.2:1b",
        base_url=OLLAMA_URL,
        temperature=0.2
    )
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=OLLAMA_URL
    )
    return llm, embeddings

llm, embeddings = load_models()

# --------------------------------------------------
# Prompt RAG (IMPORTANT : {input} et {context})
# --------------------------------------------------
template = """Tu es un assistant technique.
Utilise le contexte fourni pour répondre. Chaque extrait commence par [Source: nom_du_fichier].
Cite toujours le nom du fichier source dans ta réponse.

Contexte:
{context}

Question:
{input}

Réponse en français:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

DATA_DIR_PATH = "data/"

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

# --------------------------------------------------
# Barre latérale : indexation
# --------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    if st.button("📥 Indexer les fichiers du dossier \"data\""):
        with st.spinner("Indexation en cours..."):
            docs = read_data_dir_files()

            if not docs:
                st.warning("Aucun fichier PDF trouvé dans le dossier /data")
            else:
                # On utilise un 'set' pour éviter les doublons si un fichier a plusieurs pages
                file_names = sorted(list(set([os.path.basename(d.metadata.get('source', 'Inconnu')) for d in docs])))
                
                st.info(f"📄 {len(file_names)} fichiers détectés")
                with st.expander("Détail des fichiers trouvés"):
                    for name in file_names:
                        st.write(f"- {name}")
                # ---------------------------------------------------------------

                with st.spinner("Création des vecteurs (Embeddings)..."):
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=100
                    )
                    splits = splitter.split_documents(docs)

                    st.session_state.vectorstore = Chroma.from_documents(
                        documents=splits,
                        embedding=embeddings,
                        persist_directory="./chroma_db"
                    )

                # FORCE le rechargement pour être sûr que Streamlit utilise la version disque
                st.session_state.vectorstore = Chroma(
                    persist_directory="./chroma_db",
                    embedding_function=embeddings
                )
                
                st.success(f"✅ {len(splits)} morceaux indexés avec succès !")

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
if prompt := st.chat_input("Posez votre question sur vos documents..."):
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    if "vectorstore" not in st.session_state:
        st.error("Veuillez d'abord indexer les documents dans la barre latérale.")
    else:
        with st.chat_message("assistant"):
            retriever = st.session_state.vectorstore.as_retriever(
                search_kwargs={"k": 4}
            )
            
            # Préparation du contexte formaté avec les noms de fichiers
            # On utilise ta fonction format_docs_with_sources définie plus haut
            context_docs = retriever.invoke(prompt)
            formatted_context = format_docs_with_sources(context_docs)

            # Appel au LLM avec le contexte déjà formaté
            # On utilise directement le LLM + Prompt car on a déjà géré le contexte
            # (Plus simple et plus de contrôle que create_retrieval_chain ici)
            full_prompt = QA_CHAIN_PROMPT.format(context=formatted_context, input=prompt)
            answer = llm.invoke(full_prompt)

            st.markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
