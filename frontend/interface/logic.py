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


def run_query_pipeline(query: str, config: dict = None, model: str = None):
    """
    Run the RAG query pipeline.

    This function retrieves documents relevant to the query and generates
    an answer based on the retrieved documents.

    Args:
        query: The query to process.
        config: Configuration dictionary from the UI. If None, uses the global configuration.
        model: The generation model to use. If None, uses the model from configuration.

    Returns:
        Dictionary with the query results.
    """
    # Get configuration
    generation_config = config_manager.get_generation_config()

    # Use UI config if provided, otherwise use global config
    if config is None:
        config = {}
        retrieval_config = config_manager.get_retrieval_config()
        chromadb_config = config_manager.get_chromadb_config()
        config["collection"] = chromadb_config.get("collection_name", "default")
        config["use_query_rewrite"] = retrieval_config.get("use_query_rewrite", True)
        config["use_multi_query"] = retrieval_config.get("use_multi_query", True)
        config["use_hyde"] = retrieval_config.get("use_hyde", True)
        config["top_k"] = retrieval_config.get("top_k", 4)
        config["system_prompt"] = generation_config.get("system_prompt", "")
    # Initialize retriever
    retriever = RAGRetriever()

    # 1. Retrieve documents
    docs = retriever.retrieve(
        question=query,
    )

    # 2. Generate answer
    context = "\n\n".join(doc.page_content for doc, _ in docs)

    # Get QA prompt template from config
    qa_prompt_template = generation_config.get(
        "qa_prompt_template", 
        "<s>[INST] {system_prompt}\n\nContexte :\n{context}\n\nQuestion : {question} [/INST]"
    )

    qa_prompt = PromptTemplate.from_template(qa_prompt_template)

    full_prompt = qa_prompt.format(
        system_prompt=config.get("system_prompt", generation_config.get("system_prompt", "")),
        context=context,
        question=query
    )
    print("🧾 Prompt envoyé :\n", full_prompt)

    response = retriever.llm.invoke(full_prompt)

    return {
        "source_documents": docs,  # [(doc, score)]
        "query_embedding": retriever.embedder.embed_query(query),
        "doc_embeddings": [
            retriever.embedder.embed_query(doc.page_content) for doc, _ in docs
        ],
        "similarity_scores": [score for _, score in docs],
        "result": response,
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
