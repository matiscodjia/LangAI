import streamlit as st

def sidebar_controls():
    with st.sidebar:
        st.header("🔧 Configuration")

        collection = st.text_input("Collection Name", value="documents_collection")
        visualization = st.text_input("Visualization method", value="pca")
        k = st.slider("Top K Results", 1, 20, 4)

        filter_key = st.text_input("Metadata Key (optional)")
        filter_value = st.text_input("Metadata Value (optional)")

        st.markdown("---")
        st.subheader("📦 Retrieval Options")
        use_query_rewrite = st.checkbox("Use Query Rewriting", value=True)
        use_multi_query = st.checkbox("Use Multi-querying", value=True)
        use_hyde = st.checkbox("Use HyDE", value=True)
        use_rerank = st.checkbox("Use Reranking", value=True)
        st.markdown("---")
    return {
        "collection": collection,
        "top_k": k,
        "filter_key": filter_key,
        "filter_value": filter_value,
        "visualization":visualization,
        "use_query_rewrite": use_query_rewrite,
        "use_multi_query": use_multi_query,
        "use_hyde": use_hyde,
        "use_rerank": use_rerank
    }
