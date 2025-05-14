import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import streamlit as st

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from backend.RagCore.Retrieving.retriever import RAGRetriever

QA_PROMPT = PromptTemplate.from_template(
    "Here is some documentary context:\n\n{context}\n\nBased only on this context, answer the following question: {question}"
)

RESULTS_DIR = "evaluations"
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_query_pipeline(query: str, config: dict):
    retriever = RAGRetriever(
        collection_name=config["collection"]
    )

    # 1. Retrieve
    docs = retriever.retrieve(
        question=query,
        use_rewrite=config["use_query_rewrite"],
        use_multi_query=config["use_multi_query"],
        use_hyde=config["use_hyde"],
        top_k=config["top_k"]
    )

    # 2. Generate answer
    context = "\n\n".join(doc.page_content for doc, _ in docs)
    generation_chain = QA_PROMPT | retriever.llm
    response = generation_chain.invoke({"context": context, "question": query})

    return {
        "source_documents": docs,  # [(doc, score)]
        "query_embedding": retriever.embedder.embed_query(query),
        "doc_embeddings": [retriever.embedder.embed_query(doc.page_content) for doc, _ in docs],
        "similarity_scores": [score for _, score in docs],
        "result": response
    }


def visualize_embeddings(query_vec, doc_vecs, method="pca", n_components=2, scores=None):
    all_vecs = np.vstack([query_vec] + doc_vecs)
    labels = ["query"] + [f"doc_{i}" for i in range(len(doc_vecs))]

    if method == "pca":
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(all_vecs)
        print(f"✅ PCA: {n_components} components explain {np.sum(pca.explained_variance_ratio_):.2f} variance")

    elif method == "tsne":
        n_samples = len(all_vecs)
        perplexity = min(30, max(2, n_samples - 1))
        tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=200, random_state=42)
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
        size="score"
    )
    fig.update_layout(
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        plot_bgcolor='white'
    )
    return fig


def display_retrieved_docs(docs):
    for i, (doc, score) in enumerate(docs):
        color = "🟩" if score > 0.8 else "🟨" if score > 0.6 else "🟥"
        st.markdown(f"### {color} Document {i+1} — Similarity: {score:.3f}")
        st.text(doc.page_content[:1000])  # Truncate for readability
        # st.json(doc.metadata)


def save_evaluation(config: dict, query: str, rating: str, comment: str):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    data = {
        "timestamp": run_id,
        "query": query,
        "collection": config["collection"],
        "top_k": config["top_k"],
        "use_query_rewrite": config["use_query_rewrite"],
        "use_multi_query": config["use_multi_query"],
        "use_hyde": config["use_hyde"],
        "use_rerank": config["use_rerank"],
        "rating": rating,
        "comment": comment
    }
    df = pd.DataFrame([data])
    filename = os.path.join(RESULTS_DIR, "evaluations_log.csv")
    df.to_csv(filename, mode="a", header=not os.path.exists(filename), index=False)


def render_query_results(query, result, config):
    if not result["source_documents"]:
        st.warning("❗ No documents were retrieved for this query with the selected filters.")
        return

    st.markdown("## 🧠 Generated Answer")
    st.write(result["result"])

    st.subheader("🔍 Retrieved Documents")
    display_retrieved_docs(result["source_documents"])

    st.markdown("## 📊 Embedding Space Visualization")
    if not result["doc_embeddings"]:
        st.warning("📉 No documents available for embedding visualization.")
    else:
        fig = visualize_embeddings(
            query_vec=result["query_embedding"],
            doc_vecs=result["doc_embeddings"],
            method=config["visualization"],
            scores=result["similarity_scores"]
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("## 🧪 Human Evaluation")

    if "rating" not in st.session_state:
        st.session_state.rating = "Not at all"
    if "comment" not in st.session_state:
        st.session_state.comment = ""

    def update_rating():
        st.session_state["rating_selected"] = st.session_state["rating"]

    def update_comment():
        st.session_state["comment_written"] = st.session_state["comment"]

    st.radio(
        "How relevant was the answer?",
        ["Not at all", "Somewhat", "Mostly", "Perfect"],
        key="rating",
        on_change=update_rating
    )

    st.text_area(
        "Comments or observations",
        key="comment",
        on_change=update_comment
    )

    if st.button("Submit Evaluation"):
        if "last_query" in st.session_state and "last_config" in st.session_state:
            save_evaluation(
                config=st.session_state["last_config"],
                query=st.session_state["last_query"],
                rating=st.session_state.get("rating_selected", st.session_state["rating"]),
                comment=st.session_state.get("comment_written", st.session_state["comment"])
            )
            st.success("✅ Evaluation saved.")
        else:
            st.warning("⚠️ No query has been executed yet.")