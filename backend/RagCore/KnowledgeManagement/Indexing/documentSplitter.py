import logging
from backend.RagCore.KnowledgeManagement.Indexing.duckdbManager import DuckDBManager, read_metadata_from_duckdb

from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter

EMBEDDING_MODEL = "nomic-embed-text"


class DocumentSplitter:
    def __init__(self, embedding_model: str = EMBEDDING_MODEL):
        self.embedding_model = embedding_model

    def load_data(self):
        """Load texts and metadata from the DuckDB database."""
        manager = DuckDBManager()
        df = manager.read_metadata_from_duckdb()
        texts = df["texte"].tolist()
        metadata_list = df.drop(columns=["texte"]).to_dict(orient="records")
        return texts, metadata_list

    def _get_splitter(self, mode: str):
        """Return the appropriate splitter instance based on the mode."""
        if mode == "semantic":
            if not self.embedding_model:
                raise ValueError("Embedding model is not defined.")
            embeddings = OllamaEmbeddings(model=self.embedding_model)
            return SemanticChunker(embeddings=embeddings)

        elif mode == "recursive":
            return RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200,
                separators=["\n\n", "\n", " "]
            )

        elif mode == "token":
            return CharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base",
                chunk_size=1500
            )

        raise ValueError(f"Unknown splitting mode: {mode}")

    def split(self, mode: str = "semantic", verbose: bool = True):
        """
        Perform text splitting using the specified mode.
        :param mode: One of ["semantic", "recursive", "token"]
        :param verbose: Whether to print the number of generated documents.
        :return: List of LangChain documents
        """
        texts, metadatas = self.load_data()
        documents = []

        splitter = self._get_splitter(mode)

        for text, meta in zip(texts, metadatas):
            try:
                docs = splitter.create_documents([text], metadatas=[meta])
                documents.extend(docs)
            except Exception as e:
                logging.error("Error during %s splitting: %s", mode, e)

        if verbose:
            print(f"Number of documents ({mode.capitalize()}): {len(documents)}")

        return documents

    def split_semantic(self, verbose=True):
        return self.split(mode="semantic", verbose=verbose)

    def split_recursive(self, verbose=True):
        return self.split(mode="recursive", verbose=verbose)

    def split_token(self, verbose=True):
        return self.split(mode="token", verbose=verbose)