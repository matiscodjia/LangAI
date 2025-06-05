import os
import requests
from tqdm import tqdm
from typing import List, Union
from pathlib import Path
from dotenv import load_dotenv
from abc import ABC, abstractmethod

from langchain.schema import Document
from chromadb import PersistentClient

from backend.RagCore.Utils.pathProvider import PathProvider

# Optional backends
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaEmbeddings
from huggingface_hub import model_info
import ollama

load_dotenv()


# === Embedding interface ===
class EmbeddingBackend(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        pass


# === Utilitaire : lister les modèles Ollama disponibles ===
def get_available_models():
    try:
        models = ollama.list()
        return [model["model"] for model in models["models"]]
    except Exception as e:
        print(f"[⚠️] Could not connect to Ollama: {e}")
        return []


# === OllamaEmbedder utilisant langchain_ollama ===
class OllamaEmbedder(EmbeddingBackend):
    def __init__(self, model_name: str = "nomic-embed-text:latest"):
        self.model_name = model_name
        self.available = self._check_model()
        if self.available:
            self.embedder = OllamaEmbeddings(model=self.model_name)
        else:
            self.embedder = None

    def _check_model(self) -> bool:
        return self.model_name in get_available_models()

    def get_model(self):
        return self.embedder

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not self.embedder:
            print(
                f"Modèle Ollama '{self.model_name}' indisponible — fallback null embedding."
            )
            return [[0.0] * 768 for _ in texts]
        try:
            return self.embedder.embed_documents(texts)
        except Exception as e:
            print(f"Échec embedding Ollama : {e}")
            return [[0.0] * 768 for _ in texts]


# === HFEmbedder ===
class HFEmbedder(EmbeddingBackend):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = self._charger_modele()

    def _verifier_modele(self) -> bool:
        try:
            model_info(self.model_name)
            return True
        except Exception as e:
            print(f"Vérification modèle HF '{self.model_name}' échouée : {e}")
            return False

    def _charger_modele(self):
        if not self._verifier_modele():
            return None
        try:
            return SentenceTransformer(self.model_name).to(self.device)
        except Exception as e:
            print(f"Chargement modèle HF '{self.model_name}' échoué : {e}")
            return None

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not self.model:
            print("Modèle HF non disponible — fallback null embedding.")
            return [[0.0] * 384 for _ in texts]
        try:
            return self.model.encode(texts, normalize_embeddings=True).tolist()
        except Exception as e:
            print(f"[⚠️] Embedding HF échoué : {e}")
            return [[0.0] * 384 for _ in texts]


# === Factory function ===
def get_embedder(
    backend: str, model_name: str, device: str = "cpu"
) -> EmbeddingBackend:
    if backend == "ollama":
        return OllamaEmbedder(model_name)
    elif backend == "hf":
        return HFEmbedder(model_name, device)
    else:
        raise ValueError(f"Backend non supporté : {backend}")


# === ChromaEmbedder class ===
# loading_documents.py (ou dans le fichier où se trouve ChromaEmbedder)

from tqdm.auto import tqdm        # vous pouvez conserver tqdm pour la console
from typing import Callable, List
from langchain.schema import Document

class ChromaEmbedder:
    def __init__(
        self,
        collection_name: str = "documents_collection",
        embedding_backend = None,     # votre implémentation de EmbeddingBackend
        persist_path: str = None,
    ):
        self.embedding_backend = embedding_backend
        self.path_provider = PathProvider()
        chroma_path = persist_path or self.path_provider.chroma()

        # ───> On force tenant_id="" pour désactiver le "default_tenant" de ChromaDB
        self.client = PersistentClient(path=str(chroma_path))
        self.collection = self.client.get_or_create_collection(name=collection_name)
        

    def store_documents(
        self,
        docs: List[Document],
        progress_callback: Callable[[int], None] = None,
        use_tqdm: bool = False,
    ) -> None:
        """
        Stocke une liste de `Document` dans la collection Chroma.

        :param docs: liste de Document à indexer
        :param progress_callback: fonction à appeler avec un entier 0..100 pour indiquer la progression
                                   Si None, aucune mise à jour Streamlit n’est envoyée.
        :param use_tqdm: si True, on affiche aussi une barre tqdm dans la console environnante.
        """
        total = len(docs)
        print(
            f"📦 Ajout de {total} documents dans la collection '{self.collection.name}'"
        )

        # Si use_tqdm=True, on enveloppe docs dans tqdm pour la console, sinon on boucle directement.
        iterator = tqdm(docs, desc="Embedding & Stockage", colour="green") if use_tqdm else docs

        for idx, doc in enumerate(iterator, start=1):
            try:
                # On récupère l’embedding (liste de floats) via votre backend
                embedding = self.embedding_backend.embed([doc.page_content])[0]
                # On ajoute le document à Chroma
                self.collection.add(
                    ids=[f"{idx}"],
                    embeddings=[embedding],
                    metadatas=[doc.metadata],
                    documents=[doc.page_content],
                )
            except Exception as e:
                print(f"Erreur stockage doc {idx} : {e}")

            # Appel du callback (0..100) si fourni
            if progress_callback is not None:
                percent = int(idx / total * 100)
                progress_callback(percent)