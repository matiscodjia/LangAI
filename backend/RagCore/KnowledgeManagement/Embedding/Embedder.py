import os
from tqdm import tqdm
from typing import List, Union
from pathlib import Path
from dotenv import load_dotenv

from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
from chromadb import PersistentClient

from RagCore.Utils.pathProvider import PathProvider

load_dotenv()

OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")


class ChromaEmbedderHF:
    def __init__(
        self,
        collection_name: str = "documents_collection",
        model_name: str = OLLAMA_EMBEDDING_MODEL,
        persist_path: str = None,
    ):
        """
        :param collection_name: Name of the ChromaDB collection.
        :param model_name: Hugging Face model name.
        :param persist_path: Custom path to ChromaDB folder (optional).
        """
        self.embedding_model = OllamaEmbeddings(model=model_name)

        self.path_provider = PathProvider()
        chroma_path = persist_path or self.path_provider.chroma()

        self.client = PersistentClient(path=str(chroma_path))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def store_documents(self, docs: List[Document]) -> None:
        """
        Embed and store a list of LangChain Document objects into the Chroma collection.

        :param docs: List of Document objects (page_content + metadata)
        """
        print(
            f"📦 Storing {len(docs)} documents in collection '{self.collection.name}'"
        )

        for idx, doc in enumerate(
            tqdm(docs, desc="Embedding & Storing", colour="green")
        ):
            doc_id = f"{idx}"
            try:
                embedding = self.embedding_model.embed_documents([doc.page_content])[0]
                self.collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    metadatas=[doc.metadata],
                    documents=[doc.page_content],
                )
            except Exception as e:
                print(f"[⚠️] Error storing doc {idx}: {e}")
