import streamlit as st
import yaml
import os
import sys
import logging

from backend.RagCore.Utils.configManager import ConfigManager
from frontend.interface.logic import render_query_results, run_query_pipeline
from frontend.ui.ui_backend import sidebar_controls
from frontend.loading_documents import run_loading_pipeline

sys.modules['torch.classes'] = None

log = logging.getLogger("RAGWebApp")
logging.basicConfig(level=logging.INFO, format="%(message)s")

# --- CONFIGURATION GLOBALE STREAMLIT ---
os.environ["STREAMLIT_WATCHFILE"] = "false"
st.set_page_config(layout="wide")
st.title("Dashboard RAG")

# --- INIT CONFIG ---
config_manager = ConfigManager()
default_config = config_manager.get_config()

# --- MENU DE NAVIGATION ---
page = st.sidebar.selectbox(
    "Navigation",
    ("Accueil", "Recherche RAG", "Indexation / Vectorisation")
)

# --- PAGE : ACCUEIL ---
def page_home():
    st.header("Accueil")
    st.write("Bienvenue sur l’interface RAG ! Utilisez le menu de gauche pour naviguer.")

# --- PAGE : RECHERCHE (RAG) ---
def page_rag():
    st.header("Recherche (RAG)")
    config = ConfigManager()  # ← ajout de cette ligne
    sidebar_controls(config)
    query = st.text_input("Entrez la question à poser au RAG")
    run_button = st.button("Lancer la recherche")
    model = st.selectbox(
        "Modèle de génération",
        options=config_manager.get("ui.generation_models", []),
    )
    if run_button and query:
        result = run_query_pipeline(query)
        render_query_results(result)

# --- PAGE : INDEXATION ---
def page_indexation():
    st.header("Indexation & Vectorisation")

    doc_cfg = config_manager.get_document_processing_config()
    chroma_cfg = config_manager.get_chromadb_config()
    ui_cfg = config_manager.get_ui_config()

    st.subheader("Configuration YAML")
    yaml_text = st.text_area(
        "Contenu de la configuration",
        value=yaml.dump(default_config, allow_unicode=True),
        height=200
    )

    try:
        user_params = yaml.safe_load(yaml_text)
    except Exception as e:
        st.error(f"Erreur de parsing YAML : {e}")
        user_params = default_config

    st.subheader("Options manuelles")
    data_source = st.text_input("Chemin vers les fichiers", value=doc_cfg.get("path"))
    chunking_strategy = st.selectbox("Stratégie de découpe", options=ui_cfg.get("chunking_strategies", []))
    export_split = st.checkbox("Exporter les documents", value=doc_cfg.get("export_split", True))
    export_name = st.text_input("Nom de l’export", value=doc_cfg.get("export_name", "default"))
    embedding_model = st.selectbox("Modèle d’embedding", options=ui_cfg.get("embedding_models", []))
    embedding_device = st.selectbox("Device", options=["cpu", "cuda"], index=0)
    collection_name = st.text_input("Nom de la collection", value=chroma_cfg.get("collection_name", "default"))
    advanced_metadatas = st.checkbox("Métadonnées avancées", value=doc_cfg.get("advanced_metadatas", False))

    override_config = st.checkbox("Remplacer la configuration YAML par les paramètres manuels")

    if st.button("Lancer l’indexation"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(percent: int):
            progress_bar.progress(percent)
            status_text.text(f"Progression : {percent}%")

        pipeline_params = {
            "data_source": data_source,
            "chunking_strategy": chunking_strategy,
            "export_split": export_split,
            "export_name": export_name,
            "embedding_model_name": embedding_model,
            "collection_name": collection_name,
            "advanced_metadatas": advanced_metadatas,
            "progress_callback": progress_callback,
            "use_tqdm": False,
        }

        st.info("Lancement de la pipeline...")
        with st.spinner("Traitement en cours..."):
            try:
                if override_config:
                    run_loading_pipeline(config=config_manager, config_overrides=pipeline_params)
                else:
                    run_loading_pipeline(config=config_manager)
                st.success("Indexation terminée avec succès !")
            except Exception as e:
                log.error(f"Erreur pendant l’indexation : {e}")
                st.error(f"Erreur pendant l’indexation : {e}")

# === ROUTAGE PRINCIPAL ===
if page == "Accueil":
    page_home()
elif page == "Recherche RAG":
    page_rag()
elif page == "Indexation / Vectorisation":
    page_indexation()