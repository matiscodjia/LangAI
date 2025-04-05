from chromadb import PersistentClient
import numpy as np
from tqdm import tqdm
##Pas d'écriture concurrente possible


def store_documents(docs,embedding_model, collection_name="documents_collection",path="data/chroma_db"):
    """
    Pour chaque document de la liste, calcule son embedding et l'insère dans la collection ChromaDB
    dans le namespace spécifié.
    
    :param docs: Liste d'objets Document (ayant au moins les attributs page_content et metadata)
    :param namespace: Namespace dans lequel stocker les documents (ex: "semantic_chunk")
    :param embedding_model: Objet d'embedding (par exemple, OllamaEmbeddings) avec la méthode embed_documents()
    :param collection_name: Nom de la collection dans ChromaDB
    """
    # Initialiser le client ChromaDB et récupérer ou créer la collection
    client = PersistentClient(path=path)
    collection = client.get_or_create_collection(name=collection_name)
    
    # Boucle avec tqdm pour afficher la progression
    for idx, doc in enumerate(tqdm(docs,colour="cyan")):
        # Créer un identifiant unique pour ce document dans ce namespace
        doc_id = f"{idx}"
        # Calculer l'embedding du contenu du document
        doc_embedding = embedding_model.embed_documents([doc.page_content])[0]
        # Insérer le document dans la collection en spécifiant le namespace
        collection.add(
            ids=[doc_id],
            embeddings=[doc_embedding],
            metadatas=[doc.metadata],
            documents=[doc.page_content],
        )