import streamlit as st
from chromadb import PersistentClient
from backend.RagCore.Utils.pathProvider import PathProvider
def sidebar_controls():
    # Section générale de configuration
    provider = PathProvider()
    chromaPath = provider.chroma()
    client = PersistentClient(path=str(chromaPath))
    with st.sidebar.expander("🔧 Configuration générale", expanded=True):
        try:
            collection_names = client.list_collections()  # liste de strings
        except Exception as e:
                st.sidebar.error(f"❌ Erreur lors du listing des collections : {e}")
                collection_names = []
        
        if collection_names:
            # Si au moins une collection existe, on propose un selectbox
            collection = st.selectbox(
                "Nom de la collection ChromaDB",
                options=collection_names,
                index=0,
                help="Sélectionnez la collection Chroma qui contient vos embeddings."
            )
        else:
            # Si aucune collection trouvée, on informe l'utilisateur et on propose quand même un champ texte
            st.sidebar.warning("⚠️ Aucune collection Chroma trouvée dans le dossier spécifié.")
            collection = st.text_input(
                "Nom de la nouvelle collection (sera créée à l’indexation)",
                value="default",
                help="Aucune collection existante n’a été détectée, entrez un nom."
            )
        visualization = st.selectbox(
            "Visualization method",
            options=["pca", "tsne", "umap"],
            index=0,
            help="Méthode pour réduire la dimensionnalité (affichage des embeddings)."
        )
        k = st.slider(
            "Top K Results",
            min_value=1, max_value=20, value=4,
            help="Nombre de documents à récupérer avant réinjection dans le LLM."
        )

    # Saut de ligne visuel
    st.sidebar.markdown("---")

    # Section « System prompt » et options avancées
    with st.sidebar.expander("🤖 System Prompt & Options avancées", expanded=False):
        system_prompt = st.text_area(
            "System prompt",
            value=(
                "Tu es un assistant utile qui répond aux questions en te basant sur le contexte fourni.\n\n"
                "Si la réponse ne se trouve pas dans le contexte, réponds : "
                "« Je n'ai pas assez d'informations pour répondre à cette question. »"
            ),
            height=200,
            help="Prompt système envoyé avant la question utilisateur pour guider le LLM.",
            key="system_prompt"
        )

        st.markdown("**Retrieval Options**")
        cols1, cols2 = st.columns(2)
        with cols1:
            use_query_rewrite = st.checkbox(
                "Use Query Rewriting", value=True,
                help="Réécrit la requête pour améliorer la recherche d'embeddings."
            )
            use_multi_query = st.checkbox(
                "Use Multi-querying", value=True,
                help="Génère plusieurs reformulations de la question pour creuser les résultats."
            )
        with cols2:
            use_hyde = st.checkbox(
                "Use HyDE", value=True,
                help="Ajoute des documents hypothétiques (HyDE) afin de diversifier le contexte."
            )
            use_rerank = st.checkbox(
                "Use Reranking", value=True,
                help="Réordonne les documents par similarité (cosine) après fusion."
            )

    # Section facultative : filtres de métadonnées
    st.sidebar.markdown("---")
    with st.sidebar.expander("🔎 Filtres (optionnels)", expanded=False):
        filter_key = st.text_input(
            "Metadata Key",
            value="",
            help="Clé du champ de métadonnée pour filtrer les documents (ex : 'date', 'author', …)."
        )
        filter_value = st.text_input(
            "Metadata Value",
            value="",
            help="Valeur du champ de métadonnée pour le filtrage (ex : '2024-05-01')."
        )

    return {
        "collection": collection,
        "visualization": visualization,
        "top_k": k,
        "system_prompt": system_prompt,
        "use_query_rewrite": use_query_rewrite,
        "use_multi_query": use_multi_query,
        "use_hyde": use_hyde,
        "use_rerank": use_rerank,
        "filter_key": filter_key,
        "filter_value": filter_value,
    }