import streamlit as st
import os

# Ollama
from langchain_ollama import OllamaLLM, OllamaEmbeddings

# LangChain (0.3.x – structure correcte)
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --------------------------------------------------
# Configuration Streamlit
# --------------------------------------------------
st.set_page_config(page_title="Mon IA Offline (RAG)", layout="wide")
st.title("📂 Assistant PDF Local (RAG)")

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
Réponds de manière concise en utilisant uniquement le contexte fourni.
Si la réponse n'est pas dans le contexte, dis simplement que tu ne sais pas.

Contexte:
{context}

Question:
{input}

Réponse en français:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# --------------------------------------------------
# Barre latérale : indexation
# --------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    if st.button("📥 Indexer les PDF du dossier /data"):
        with st.spinner("Indexation en cours..."):
            loader = DirectoryLoader(
                "data/",
                glob="*.pdf",
                loader_cls=PyPDFLoader
            )
            docs = loader.load()

            if not docs:
                st.warning("Aucun fichier PDF trouvé dans le dossier /data")
            else:
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

                st.success(f"✅ {len(splits)} morceaux indexés")

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

            document_chain = create_stuff_documents_chain(
                llm=llm,
                prompt=QA_CHAIN_PROMPT
            )

            qa_chain = create_retrieval_chain(
                retriever=retriever,
                combine_docs_chain=document_chain
            )

            response = qa_chain.invoke({"input": prompt})
            answer = response["answer"]

            st.markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
