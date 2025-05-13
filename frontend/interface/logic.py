import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
import os
from datetime import datetime
from langchain_core.runnables import RunnableLambda

from modules.retrieving.retrieving import (
    enhanced_retrieve,
    get_retriever,
    OllamaLLM
)
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_ollama import OllamaLLM

QA_PROMPT = PromptTemplate.from_template(
    "Voici un contexte documentaire :\n\n{context}\n\nEn te basant uniquement sur ce contexte, réponds à la question suivante : {question}"
)

def run_query_pipeline(query: str, config: dict):
    llm = OllamaLLM(model="llama3.2")

    metadata_filter = {
    config["filter_key"]: config["filter_value"]
    } if config["filter_key"] and config["filter_value"] else None

    retriever = get_retriever(config["collection"], k=config["top_k"], metadata_filter=metadata_filter)
    embedder = OllamaEmbeddings(model="nomic-embed-text")

    # 1. Retrieve
    docs = enhanced_retrieve(
        llm=llm,
        retriever=retriever,
        question=query,
        embedder=embedder,
        use_query_rewrite=config["use_query_rewrite"],
        use_multi_query=config["use_multi_query"],
        use_hyde=config["use_hyde"],
        top_k=config["top_k"]
    )

    # 2. Generate answer
    context = "\n\n".join(doc.page_content for doc in docs)
    generation_chain = LLMChain(llm=llm, prompt=QA_PROMPT)
    response = generation_chain.invoke({"context": context, "question": query})["text"]
    #generation_chain = QA_PROMPT | llm
    #response = generation_chain.invoke({"context": ..., "question": ...})

    return {
        "source_documents": docs,
        "query_embedding": embedder.embed_query(query),
        "doc_embeddings": [embedder.embed_query(doc.page_content) for doc in docs],
        "result": response  # ✅ la réponse générée par le LLM
    }

RESULTS_DIR = "evaluations"
os.makedirs(RESULTS_DIR, exist_ok=True)

def visualize_embeddings(query_vec, doc_vecs):
    pca = PCA(n_components=2)
    all_vecs = np.vstack([query_vec] + doc_vecs)
    reduced = pca.fit_transform(all_vecs)
    df = pd.DataFrame(reduced, columns=pd.Index(["x", "y"]))
    df["label"] = ["query"] + [f"doc_{i}" for i in range(len(doc_vecs))]
    fig = px.scatter(df, x="x", y="y", color="label", title="Embedding Visualization")
    return fig

def display_retrieved_docs(docs):
    for i, doc in enumerate(docs):
        st.markdown(f"### Document {i+1}")
        st.text(doc.page_content[:1000])  # truncate
        st.json(doc.metadata)

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
        st.warning("❗ Aucun document correspondant n’a été trouvé avec ce filtre.")
        return  # On sort pour éviter tout crash ensuite
    st.markdown("## 🧠 Generated Answer")
    st.write(result["result"])

    st.subheader("🔍 Retrieved Results")
    st.markdown("## 📂 Retrieved Documents")
    display_retrieved_docs(result["source_documents"])

    st.markdown("## 📊 Embedding Space Visualization")
    if len(result["doc_embeddings"]) == 0:
        st.warning("📉 Pas de documents pour la visualisation des embeddings.")
    else:
        fig = visualize_embeddings(result["query_embedding"], result["doc_embeddings"])
        st.plotly_chart(fig, use_container_width=True)
 
    st.markdown("## 🧪 Human Evaluation")

    # Init session state
    if "rating" not in st.session_state:
        st.session_state.rating = "Not at all"
    if "comment" not in st.session_state:
        st.session_state.comment = ""

    # Controlled form
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
        "Comments or Observations",
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
            st.warning("⚠️ No query has been run yet.")
