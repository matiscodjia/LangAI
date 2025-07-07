import streamlit as st

from backend.RagCore.Utils.pathProvider import PathProvider
from backend.RagCore.Utils.configManager import ConfigManager

# Initialize configuration manager
config_manager = ConfigManager()

def sidebar_controls(config):
    # Section générale de configuration
    provider = PathProvider()
    chromapath = provider.chroma()

    with st.sidebar.expander("Configuration générale", expanded=True):
        try:
            from chromadb import PersistentClient
            client = PersistentClient(path=str(chromapath))
            collection_names = [col.name for col in client.list_collections()]
        except Exception as e:
            st.sidebar.error(f"❌ Erreur lors du listing des collections : {e}")
            collection_names = []

        if collection_names:
            collection = st.selectbox(
                "Nom de la collection ChromaDB",
                options=collection_names,
                index=0,
                help="Sélectionnez la collection Chroma qui contient vos embeddings."
            )
        else:
            st.sidebar.warning("⚠️ Aucune collection Chroma trouvée dans le dossier spécifié.")
            collection = st.text_input(
                "Nom de la nouvelle collection (sera créée à l’indexation)",
                value="default",
                help="Aucune collection existante n’a été détectée, entrez un nom."
            )

        # Embedding model
        default_model = config.get("embedding_model", "all-mpnet-base-v2")
        model_options = ["all-mpnet-base-v2", "gte-small", "sentence-transformers/all-MiniLM-L6-v2", "antoinelouis/french-gte-multilingual-base"]
        model_index = model_options.index(default_model) if default_model in model_options else 0

        selected_model = st.selectbox(
            "Modèle d'embedding",
            options=model_options,
            index=model_index,
            help="Modèle utilisé pour générer les embeddings."
        )

        # Update config
        config._config["embedding_model"] = selected_model
        config._config["collection_name"] = collection

        # Top-K
        top_k = st.slider(
            "Top K Documents",
            min_value=1, max_value=20,
            value=config.get("top_k", 3),
            help="Nombre de documents à récupérer après la recherche."
        )
        config._config["top_k"] = top_k

    # System Prompt
    st.sidebar.markdown("---")
    with st.sidebar.expander("System Prompt & Options avancées", expanded=False):
        default_prompt = config.get("system_prompt", "Tu es un assistant utile qui répond aux questions...")
        system_prompt = st.text_area(
            "System prompt",
            value=default_prompt,
            height=200,
            help="Prompt système envoyé avant la question utilisateur pour guider le LLM."
        )
        config._config["system_prompt"] = system_prompt

    return {
        "collection": collection,
        "embedding_model": selected_model,
        "top_k": top_k,
        "system_prompt": system_prompt,
    }