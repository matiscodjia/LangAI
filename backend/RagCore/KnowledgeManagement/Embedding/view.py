import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
from langchain_ollama.embeddings import OllamaEmbeddings
from typing import List
from langchain.schema import Document

embedding_model = OllamaEmbeddings(model="nomic-embed-text")


def get_query_embedding(query: str) -> List[float]:
    """
    Transforme une requête en vecteur d'embedding.
    """
    return embedding_model.embed_query(query)


def get_doc_embeddings(documents: List[Document]) -> List[List[float]]:
    """
    Transforme une liste de documents en liste de vecteurs d'embeddings.
    Utilise uniquement le contenu du document.
    """
    return [embedding_model.embed_query(doc.page_content) for doc in documents]


def visualize_embeddings(query_vec, doc_vecs):
    pca = PCA(n_components=2)
    all_vecs = np.vstack([query_vec] + doc_vecs)
    reduced = pca.fit_transform(all_vecs)
    df = pd.DataFrame(reduced, columns=pd.Index(["x", "y"]))
    df["label"] = ["query"] + [f"doc_{i}" for i in range(len(doc_vecs))]
    fig = px.scatter(df, x="x", y="y", color="label", title="Embedding Visualization")
    return fig
