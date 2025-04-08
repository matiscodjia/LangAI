import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px

from modules.retrieving.retrieving import get_qa_chain
from modules.embedding.view import get_query_embedding, get_doc_embeddings

# === Streamlit UI ===
st.set_page_config(layout="wide")
st.title("🧠 RAG Debug Dashboard")

with st.sidebar:
    collection = st.text_input("Collection Name", value="default")
    k = st.slider("Top K Results", 1, 20, 4)
    filter_key = st.text_input("Metadata Key (optional)")
    filter_value = st.text_input("Metadata Value (optional)")
    run_button = st.button("Run Query")

query = st.text_input("Enter your query:")

def build_filter(key, value):
    return {key: value} if key and value else None

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
        st.text(doc.page_content[:1000])
        st.json(doc.metadata)

if run_button and query:
    st.subheader("🔍 Retrieval & QA")
    metadata_filter = build_filter(filter_key, filter_value)
    chain = get_qa_chain(collection, k)
    result = chain({"query": query})

    st.markdown("## 🧾 Generated Answer")
    st.write(result["result"])

    st.markdown("## 📂 Retrieved Documents")
    docs = result["source_documents"]
    display_retrieved_docs(docs)

    with st.expander("📊 Embedding Space Visualization"):
        query_vec = get_query_embedding(query)
        doc_vecs = get_doc_embeddings(docs)
        fig = visualize_embeddings(query_vec, doc_vecs)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("🧪 Human Evaluation"):
        rating = st.radio("How relevant was the answer?", ["Not at all", "Somewhat", "Mostly", "Perfect"])
        comment = st.text_area("Comments or Observations")
        if st.button("Submit Evaluation"):
            st.success("Evaluation submitted. We won't ignore it. Probably.")

if __name__ == "__main__":
    st.write("App loaded successfully. Enter a query and hit run.")

