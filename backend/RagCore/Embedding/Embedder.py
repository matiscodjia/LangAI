from typing import List
from chromadb import PersistentClient

from backend.RagCore.Utils.configManager import ConfigManager
from backend.RagCore.Utils.pathProvider import PathProvider
from langchain.schema import Document
import time
import logging

# Load environment variables
# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class ChromaEmbedder:
    """
    ChromaEmbedder handles embedding and storing documents in a ChromaDB collection.
    """
    def __init__(
        self
    ):
        config_manager = ConfigManager()
        self.model = config_manager.get_embedder()
        self.path_provider = PathProvider()
        collection_name = config_manager.get_collection_name()
        chroma_path = config_manager.get_chroma_path()
        logger.info(f"→ Chroma se stocke dans : {chroma_path.resolve()}")
        self.client = PersistentClient(path=str(chroma_path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Initialized ChromaEmbedder with collection '{collection_name}' at '{chroma_path}'")

    def store_documents(
        self,
        docs: List[Document],
    ) -> None:
        total = len(docs)

        try:
            texts = [doc.page_content for doc in docs]
            metadata = [doc.metadata for doc in docs]
            ids = [str(i) for i in range(1, total + 1)]

            # Batch embedding
            start = time.time()
            embeddings = self.model.embed_documents(texts)
            duration = time.time() - start
            logger.info(f"Embedding completed in {duration:.2f} seconds for {total} documents")

            # Store in ChromaDB
            start = time.time()
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadata,
                documents=texts,
            )
            duration = time.time() - start
            logger.info(f"Added {total} documents to collection '{self.collection.name}' in {duration:.2f} seconds")
        except Exception as e:
            logger.error(f"Error during batch addition: {e}", exc_info=True)