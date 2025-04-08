import logging
from modules.data_managing.read import read_metadata_from_duckdb
# Import des splitters et modèles
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
# Configuration de base
EMBEDDING_MODEL="nomic-embed-text"

def sequential_split_documents(texts, metadatas, mode) -> list:
    """
    Applique le splitting de manière séquentielle selon le mode spécifié ("semantic", "recursive" ou "token").
    :param texts: Liste des textes.
    :param metadatas: Liste des métadonnées correspondantes.
    :param mode: Mode de splitting à utiliser.
    :return: Liste de documents générés.
    """
    documents = []
    for text, meta in zip(texts, metadatas):
        if mode == "semantic":
            if not EMBEDDING_MODEL:
                print("Modèle de vectorisation non défini")
                return []
            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
            splitter = SemanticChunker(embeddings=embeddings)
        elif mode == "recursive":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200,
                separators=["\n\n", "\n", " "]
            )
        elif mode == "token":
            splitter = CharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base", 
                chunk_size=1500
            )
        else:
            raise ValueError("Mode de splitting inconnu : {}".format(mode))
        try:
            docs = splitter.create_documents([text], metadatas=[meta])
            documents.extend(docs)
        except Exception as e:
            logging.error("Erreur lors du splitting en mode %s: %s", mode, e)
    return documents

def get_data():
    df = read_metadata_from_duckdb("data/metadata.duckdb")
    texts = df["texte"].tolist()
    metadata_list = df.drop(columns=["texte"]).to_dict(orient="records")
    return texts, metadata_list

def semantic_split(verbose=True):
    texts, metadata_list = get_data()
    docs_semantic = sequential_split_documents(texts, metadata_list, mode="semantic")
    if verbose:
        print(f"Nombre de documents (Semantic): {len(docs_semantic)}")
    return docs_semantic

def recursive_split(verbose=True):
    texts, metadata_list = get_data()
    docs_recursive = sequential_split_documents(texts, metadata_list, mode="recursive")
    if verbose:
        print(f"Nombre de documents (Recursive): {len(docs_recursive)}")
    return docs_recursive


def base_split(verbose=True):
    texts, metadata_list = get_data()
    docs_base = sequential_split_documents(texts, metadata_list, mode="token")
    if verbose:
        print(f"Nombre de documents (Token): {len(docs_base)}")
    return docs_base


