import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import streamlit as st

from langchain.prompts import PromptTemplate

from backend.RagCore.Retrieving.retriever import RAGRetriever
from backend.RagCore.Utils.configManager import ConfigManager

# Initialize configuration manager
config_manager = ConfigManager()


def run_query_pipeline(query: str):
    """
    Pipeline RAG simplifié : récupération + génération via RAGRetriever.
    """
    retriever = RAGRetriever()

    # Récupération des documents
    docs = retriever.retrieve(query)

    # Génération de réponse basée sur les documents
    answer = retriever.answer(query)

    return {
        "result": answer,  # chaîne de texte directement affichable
        "source_documents": docs,
        "query_embedding": retriever.embedder.embed_query(query),
        "doc_embeddings": [retriever.embedder.embed_query(doc.page_content) for doc, _ in docs],
        "similarity_scores": [score for _, score in docs],
    }


def visualize_embeddings(
    query_vec, doc_vecs, method=None, n_components=None, scores=None
):
    """
    Visualize embeddings in a lower-dimensional space.

    Args:
        query_vec: The query embedding vector.
        doc_vecs: List of document embedding vectors.
        method: Dimensionality reduction method (pca, tsne).
        n_components: Number of components for dimensionality reduction.
        scores: List of similarity scores for the documents.

    Returns:
        Plotly figure with the visualization.
    """
    # Get visualization configuration
    viz_config = config_manager.get_visualization_config()

    # Use parameters if provided, otherwise use config values
    method = method or viz_config.get("method", "pca")
    n_components = n_components or viz_config.get("n_components", 2)
    random_seed = viz_config.get("random_seed", 42)
    all_vecs = np.vstack([query_vec] + doc_vecs)
    labels = ["query"] + [f"doc_{i}" for i in range(len(doc_vecs))]

    if method == "pca":
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(all_vecs)
        print(
            f"PCA: {n_components} components explain {np.sum(pca.explained_variance_ratio_):.2f} variance"
        )

    elif method == "tsne":
        n_samples = len(all_vecs)
        viz_config = config_manager.get_visualization_config()

        # Get t-SNE parameters from config
        tsne_perplexity = viz_config.get("tsne_perplexity", 0)
        tsne_learning_rate = viz_config.get("tsne_learning_rate", 200)

        # If perplexity is 0, calculate it automatically
        if tsne_perplexity == 0:
            perplexity = min(30, max(2, n_samples - 1))
        else:
            perplexity = tsne_perplexity

        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=tsne_learning_rate,
            random_state=random_seed,
        )
        reduced = tsne.fit_transform(all_vecs)

    else:
        raise ValueError("Invalid method: choose 'pca' or 'tsne'")

    cols = [f"dim_{i}" for i in range(reduced.shape[1])]
    df = pd.DataFrame(reduced, columns=cols)
    df["label"] = labels
    df["score"] = [1.0] + (scores or [0.0] * len(doc_vecs))

    fig = px.scatter(
        df,
        x=cols[0],
        y=cols[1],
        color="score",
        hover_name="label",
        title=f"{method.upper()} Embedding Visualization",
        color_continuous_scale="Viridis",
        size="score",
    )
    fig.update_layout(
        xaxis=dict(showgrid=True), yaxis=dict(showgrid=True), plot_bgcolor="white"
    )
    return fig


def display_retrieved_docs(docs):
    for i, (doc, score) in enumerate(docs):
        color = "🟩" if score >= .8 else "🟨" if score > 0.6 else "🟥"
        st.markdown(f"### {color} Document {i+1} — Similarity: {score:.3f}")
        st.text(doc.page_content[:1000])  # Truncate for readability
        # st.json(doc.metadata)


def render_query_results(result):
    if not result["source_documents"]:
        st.warning(
            "No documents were retrieved for this query with the selected filters."
        )
        return

    st.markdown("## Generated Answer")
    st.write(result["result"])

    st.subheader("🔍 Retrieved Documents")
    display_retrieved_docs(result["source_documents"])
