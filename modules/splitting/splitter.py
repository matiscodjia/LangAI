import logging

# Import des splitters et modèles
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter

# Configuration de base
EMBEDDING_MODEL = "nomic-embed-text"

def sequential_split_documents(texts, metadatas, mode):
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
