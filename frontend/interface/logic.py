import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import streamlit as st

from langchain.prompts import PromptTemplate


from backend.RagCore.Retrieving.retriever import RAGRetriever


def run_query_pipeline(query: str, config: dict, model):
    retriever = RAGRetriever(collection_name=config["collection"], gen_model=model)
    # 1. Retrieve
    docs = retriever.retrieve(
        question=query,
        use_rewrite=config["use_query_rewrite"],
        use_multi_query=config["use_multi_query"],
        use_hyde=config["use_hyde"],
        top_k=config["top_k"],
    )

    # 2. Generate answer
    context = "\n\n".join(doc.page_content for doc, _ in docs)
    QA_PROMPT = PromptTemplate.from_template(
    "<s>[INST] {system_prompt}\n\nContexte :\n{context}\n\nQuestion : {question} [/INST]"
)
    full_prompt = QA_PROMPT.format(
    system_prompt=config["system_prompt"],
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
    query_vec, doc_vecs, method="pca", n_components=2, scores=None
):
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
        perplexity = min(30, max(2, n_samples - 1))
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=200,
            random_state=42,
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
        color = "🟩" if score > 0.8 else "🟨" if score > 0.6 else "🟥"
        st.markdown(f"### {color} Document {i+1} — Similarity: {score:.3f}")
        st.text(doc.page_content[:1000])  # Truncate for readability
        # st.json(doc.metadata)


def render_query_results(result, config):
    if not result["source_documents"]:
        st.warning(
            "❗ No documents were retrieved for this query with the selected filters."
        )
        return

    st.markdown("## Generated Answer")
    st.write(result["result"])

    st.subheader("🔍 Retrieved Documents")
    display_retrieved_docs(result["source_documents"])

    """st.markdown("## 📊 Embedding Space Visualization")
    if not result["doc_embeddings"]:
        st.warning("📉 No documents available for embedding visualization.")
    else:
        fig = visualize_embeddings(
            query_vec=result["query_embedding"],
            doc_vecs=result["doc_embeddings"],
            method=config["visualization"],
            scores=result["similarity_scores"],
        )
        st.plotly_chart(fig, use_container_width=True)"""