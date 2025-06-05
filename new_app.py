# web_app.py
import streamlit as st
from pathlib import Path
import yaml
import os

# Importe vos modules backend existants
from backend.RagCore.Utils.pathProvider import PathProvider
from frontend.interface.logic import render_query_results, run_query_pipeline
from frontend.ui.ui_backend import sidebar_controls

# Importe votre code d’indexation
from loading_documents import run_loading_pipeline, load_config
import sys
import torch

sys.modules['torch.classes'] = None

# --- CONFIGURATION GLOBALE STREAMLIT ---
os.environ["STREAMLIT_WATCHFILE"] = "false"
st.set_page_config(layout="wide")
st.title("Dashboard RAG")

# --- MENU DE NAVIGATION MANUEL ---
page = st.sidebar.selectbox(
    "Navigation",
    ("Accueil", "Recherche RAG", "Indexation / Vectorisation")
)

# --- PAGE : ACCUEIL ---
def page_home():
    st.header("🏠 Accueil")
    st.write("Bienvenue sur l’interface RAG ! Sélectionnez un menu à gauche.")

# --- PAGE : RECHERCHE (RAG) ---
def page_rag():
    st.header("🔍 Recherche (RAG)")
    config = sidebar_controls()  # vos widgets existants pour la recherche
    query = st.text_input("Entrez la question à poser au RAG")
    run_button = st.button("Run RAG")
    model = st.selectbox(
        "Modèle de génération",
        options=["deepseek-r1:7b ", "mistral:latest", "llama3.2:latest"],
        help="Nom précis du modèle Ollama uniquement"
    )
    if run_button and query:
        result = run_query_pipeline(query, config, model)
        render_query_results(result, config)

# --- PAGE : INDEXATION / VECTORIZATION ---
def page_indexation():
    st.header("⚙️ Indexation & Vectorisation (Offline)")

    # 1) Chargez la config par défaut du fichier YAML (si vous en avez un) :
    path_provider = PathProvider()
    config_path = path_provider.config_path()
    default_params = load_config(config_path)

    st.subheader("1. Charger la configuration depuis YAML (optionnel)")
    # Montre le contenu du YAML, et on peut le rééditer si on veut
    yaml_text = st.text_area(
        "Contenu du fichier default.yaml", 
        value=yaml.dump(default_params, allow_unicode=True),
        height=200
    )

    # 2) Parsez ou utilisez la config yaml_text si l’utilisateur a modifié quelque chose
    try:
        user_params = yaml.safe_load(yaml_text)
    except Exception as e:
        st.error(f"Erreur de parsing YAML : {e}")
        user_params = default_params

    st.markdown("---")
    st.subheader("2. Options manuelles (remplacent ou complètent le YAML)")
    data_source = st.text_input(
        "📂 Chemin vers les fichiers sources (raw_data)",
        value=user_params.get("data_source", str(path_provider.raw_data())),
        help="Répertoire ou fichier unique à indexer"
    )
    chunking_strategy = st.selectbox(
        "✂️ Stratégie de découpe (ex : 'recursive', 'regex')",
        options=["semantic","strategy_0","strategy_1","strategy_2","strategy_3","strategy_4","strategy_pp"],
        help="Nom de la stratégie utilisée par DocumentSplitter"
    )
    export_split = st.checkbox(
        "📤 Export JSON des chunks", 
        value=user_params.get("export_split", True),
        help="Si coché, on exporte les documents fragmentés en JSON"
    )
    export_name = st.text_input(
        "💾 Nom du fichier d’export (sans extension)",
        value=user_params.get("export_name", "default"),
    )
    embedding_model = st.selectbox(
        "🤖 Modèle d’embeddings",
        options=["nomic-embed-text:latest", "sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"],
        index=0,
        help="Choisissez le modèle d’embeddings (Ollama ou HF) "
    )
    embedding_device = st.selectbox(
        "🚀 Device pour l’embedding",
        options=["cpu", "cuda"],
        index=0,
        help="Device utilisé par HuggingFace sinon 'cpu' pour Ollama"
    )
    collection_name = st.text_input(
        "📖 Nom de la collection Chroma",
        value=user_params.get("collection_name", "default"),
        help="Facultatif : nom de la collection où stocker les embeddings"
    )
    advanced_metadatas = st.checkbox(
        "🔖 Stocker métadonnées avancées", 
        value=user_params.get("advanced_metadatas", False),
        help="Ajoute des champs supplémentaires dans DuckDB"
    )

    st.markdown("---")
    # 3) Bouton pour lancer la vectorisation
    if st.button("▶️ Lancer l’indexation & vectorisation", key="idx_run"):
        # 3.1) Créer une barre de progression et un conteneur de texte
        progress_bar = st.progress(0)
        status_text = st.empty()

        # 3.2) Définir le callback qui met à jour la barre
        def progress_callback(percent: int):
            progress_bar.progress(percent)
            status_text.text(f"Progression : {percent}%")

        # 3.3) Assemblez les paramètres dans un dict, incluant le callback
        pipeline_params = {
            "data_source": data_source,
            "chroma_path": user_params.get("chroma_path", ""),  # chemin Chroma, vu dans le YAML
            "chunking_strategy": chunking_strategy,
            "export_split": export_split,
            "export_name": export_name,
            "embedding_model": embedding_model,
            "embedding_device_if_available": embedding_device,
            "collection_name": collection_name,
            "advanced_metadatas": advanced_metadatas,
            "progress_callback": progress_callback,
            "use_tqdm": False,  # ou True si vous souhaitez conserver tqdm en console
        }

        st.info("⚙️ Début de l’indexation…")
        with st.spinner("En cours…"):
            try:
                run_loading_pipeline(**pipeline_params)
                st.success("✅ Indexation & Vectorisation terminées avec succès !")
            except Exception as e:
                st.error(f"❌ Erreur pendant l’indexation : {e}")

# === BOUCLE PRINCIPALE DE NAVIGATION ===
if page == "Accueil":
    page_home()
elif page == "Recherche RAG":
    page_rag()
elif page == "Indexation / Vectorisation":
    page_indexation()