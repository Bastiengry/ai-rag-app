import streamlit as st
import os
import shutil
import time
import gc
import stat

# Ollama
from langchain_ollama import OllamaLLM, OllamaEmbeddings

# LangChain (0.3.x – structure optimisée)
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
# Prompt RAG (IMPORTANT : {multi_doc_context}, {prompt} (et {history}))
# --------------------------------------------------
template = """
### Instructions :
Réponds à la question **uniquement** en utilisant les documents et l'historique fournis ci-dessous.
Ne réponds **jamais** avec des connaissances générales ou externes.
Si les documents ne contiennent **aucune** information pertinente, réponds simplement : "Aucune information pertinente trouvée dans les documents."
Cite toujours la source.

---
HISTORIQUE DE LA CONVERSATION :
{history}

---
DOCUMENTS :
{multi_doc_context}

Question: {prompt}
Réponse (sois précis, mais n'hésite pas à reformuler ou compléter si nécessaire) :

"""


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

def prepare_docs_with_metadata(docs):
    new_docs = []
    for doc in docs:
        source_name = os.path.basename(doc.metadata.get("source", "Inconnu"))
        # On crée un nouveau Document avec metadata “source” explicitement défini
        new_doc = Document(
            page_content=doc.page_content,
            metadata={"source": source_name}
        )
        new_docs.append(new_doc)
    return new_docs

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
                docs = prepare_docs_with_metadata(docs)
                
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

                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # On crée d'abord le vectorstore VIDE ou avec le premier batch
                    vectorstore = Chroma.from_documents(
                        documents=splits[:1], # On commence avec juste le premier morceau
                        embedding=embeddings,
                        persist_directory="./chroma_db"
                    )

                    # On ajoute le reste par paquets
                    batch_size = 10
                    for i in range(1, len(splits), batch_size):
                        batch = splits[i:i + batch_size]
                        vectorstore.add_documents(documents=batch)
                        
                        # Mise à jour de la barre
                        progress = min(i / len(splits), 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"Indexation : {i}/{len(splits)} morceaux...")

                    # Sauvegarde finale de l’index sur disque (obligatoire avant reload)
                    vectorstore.persist()
                    
                    status_text.empty()
                    progress_bar.empty()

                # IMPORTANT :
                # On recharge volontairement le vectorstore depuis le disque
                # pour s'assurer que les nouveaux fichiers indexés sont bien pris en compte
                # par Streamlit (évite les états mémoire incohérents)
                st.session_state.vectorstore = Chroma(
                    persist_directory="./chroma_db",
                    embedding_function=embeddings
                )
                
                st.success(f"✅ {len(splits)} morceaux indexés avec succès !")

    if st.button("🗑️ Vider la base de données"):
        # On réinitialise la référence dans Streamlit
        if "vectorstore" in st.session_state:
            st.session_state.vectorstore = None
            gc.collect()
            time.sleep(2)

        if os.path.exists("./chroma_db"):
            try:
                # Sous Windows, parfois il faut forcer la suppression
                def remove_readonly(func, path, _):
                    os.chmod(path, stat.S_IWUSR)
                    func(path)
                shutil.rmtree("./chroma_db", onerror=remove_readonly)
                st.success("Base de données supprimée avec succès. Veuillez ré-indexer.")
                st.rerun()
            except Exception as e:
                st.error(f"Erreur lors de la suppression : {e}\n"
                         "Essayez de redémarrer l'application ou vérifiez "
                         "qu'aucun autre processus n'utilise le dossier.")

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
        
    # --- Expander pour les logs ---
    with st.expander("📝 Voir les logs/détails techniques", expanded=False):
        log_placeholder = st.empty()  # Placeholder pour les logs


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
    # Nettoyage des logs
    log_placeholder.empty()
    
    # Affichage et stockage du message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    if "vectorstore" not in st.session_state:
        st.error("Veuillez d'abord indexer les documents dans la barre latérale.")
    else:
        with st.chat_message("assistant"):
            # --- ÉTAPE RAG CLASSIQUE ---
            vectorstore = st.session_state.get("vectorstore")
            if not vectorstore:
                st.error("Vectorstore non chargé.")
                st.stop()
                
            log_placeholder.write(f"Nombre de documents dans le vectorstore: {len(vectorstore._collection.get()['ids'])}")


                
            # Récupère l'historique des questions précédentes dans la conversation en cours
            last2Mess = st.session_state.messages[-3:-1]
            history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in last2Mess])
            
            augmented_prompt = history + "\nQuestion actuelle: " + prompt

            # Utilise la recherche avec scores
            docs_and_scores = vectorstore.similarity_search_with_score(
                augmented_prompt,
                k=3
            )
            
            for doc, score in docs_and_scores:
                log_placeholder.write(f"Score: {score} - Source: {doc.metadata.get('source', 'Inconnu')}")
            
            # Applique un seuil
            THRESHOLD = 1.1  # plus bas = plus permissif
            filtered_docs = [
                doc for doc, score in docs_and_scores
                if score <= THRESHOLD  # Chroma = distance (plus petit = meilleur)
            ]
            
            if not filtered_docs:
                answer = "Désolé, je ne trouve aucune information pertinente dans les documents pour répondre à votre question."
            else:
                multi_doc_context = ""
                
                log_placeholder.write("--- DEBUG : Documents filtrés ---")
                for doc in filtered_docs:
                    log_placeholder.write(doc.metadata.get('source','Inconnu'))

                
                for idx, doc in enumerate(filtered_docs, 1):
                    source_name = os.path.basename(doc.metadata.get('source', 'Inconnu'))
                    multi_doc_context += f"Document {idx} [Source: {source_name}]:\n{doc.page_content}\n\n"
        
                # À ajouter temporairement pour déboguer
                log_placeholder.write(f"Contexte envoyé à l'IA : {multi_doc_context}")

                # QA chain llm enrichie avec les documents, le prompt et l'historique des questions
                qa_chain = QA_CHAIN_PROMPT.format(
                    multi_doc_context=multi_doc_context,
                    prompt=prompt,
                    history=history
                )
        
                # Normalisation des apostrophes
                multi_doc_context = multi_doc_context.replace("’", "'")
                qa_chain = qa_chain.replace("’", "'")
                
                
                log_placeholder.write("--- DEBUG : qa_chain ---")
                log_placeholder.code(qa_chain, language="python")
                
                with st.spinner("L'IA réfléchit..."):
                    try:
                        answer = llm.invoke(qa_chain)
                        st.markdown(answer)
                    except Exception as e:
                        st.error(f"Erreur Ollama : {e}")
                        answer = "Erreur lors de la génération."

                # Appel au LLM avec le contexte déjà formaté
                # On utilise directement le LLM + Prompt car on a déjà géré le contexte
                # (Plus simple et plus de contrôle que create_retrieval_chain ici)
                # UTILISATION DU STREAMING (Pour voir la réponse s'afficher en direct et éviter les timeouts)
                #full_answer = ""
                #placeholder = st.empty()
                #for chunk in llm.stream(full_prompt):
                #    full_answer += chunk
                #    placeholder.markdown(full_answer + "▌")
                #placeholder.markdown(full_answer) # Nettoyage du curseur final
                #answer = full_answer

            # On enregistre la réponse finale dans l'historique
            st.session_state.messages.append({"role": "assistant", "content": answer})
