import json
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain.schema import Document
import duckdb
import logging
from backend.RagCore.KnowledgeManagement.Indexing.duckdbManager import DuckDBManager
from backend.RagCore.KnowledgeManagement.Embedding.Embedder_bis import get_embedder
from langchain_experimental.text_splitter import SemanticChunker

from backend.RagCore.Utils.pathProvider import PathProvider
from typing import List

path_provider = PathProvider()


# Custom chunking strategies
def chunking_strategy_0_splitter(chunk_size):
    return RecursiveCharacterTextSplitter(
        separators=["\nM\\.", "\n{2,}", "\n"],
        chunk_size=chunk_size,
        chunk_overlap=1,
        length_function=len,
        is_separator_regex=True,
    )


def chunking_strategy_1_splitter():
    return RecursiveCharacterTextSplitter(
        separators=['\n+[a-z]{0,2}[A-ZÉÈÀÊÔ \-\'0-9\—\.]{8,}?.*\n',"[0-9\-\—\. ]{3,}[a-z\&\'\&]*[A-ZÉÈÊÀÔ\-\.\—\°\:\; ']{10,}.*\n*"],
        chunk_size=1,
        chunk_overlap=1,
        length_function=len,
        is_separator_regex=True,
    )


class DualPassSplitter(RecursiveCharacterTextSplitter):
    def __init__(
        self, primary_separators: list[str], secondary_separators: list[str], **kwargs
    ):
        # Appel de la super classe avec les séparateurs secondaires
        super().__init__(separators=secondary_separators, **kwargs)

        # Splitter primaire indépendant
        self.primary_splitter = RecursiveCharacterTextSplitter(
            separators=primary_separators,
            chunk_size=1,
            chunk_overlap=0,
            length_function=kwargs.get("length_function", len),
            is_separator_regex=True,
        )

    def create_documents(
        self, texts: list[str], metadatas: list[dict] = None
    ) -> list[Document]:
        all_docs = []

        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas else {}
            # Premier découpage
            primary_chunks = self.primary_splitter.create_documents([text])
            # Deuxième découpage
            refined_chunks = self.create_documents_from_documents(primary_chunks, meta)
            all_docs.extend(refined_chunks)

        return all_docs

    def create_documents_from_documents(
        self, documents: list[Document], metadata: dict
    ) -> list[Document]:
        texts = [doc.page_content for doc in documents]
        return super().create_documents(texts, metadatas=[metadata] * len(texts))


class DocumentSplitter:
    def __init__(self,embedding_model):
        self.embedding_model = OllamaEmbeddings(model=embedding_model)

    def load_data(self):
        manager = DuckDBManager()
        con = duckdb.connect(manager.db_path)
        df = con.execute(
            "SELECT * FROM documents WHERE is_already_splitted = False"
        ).fetchdf()
        con.close()
        texts = df["texte"].tolist()
        metadata_list = df.drop(columns=["texte"]).to_dict(orient="records")
        return texts, metadata_list

    def _get_splitter(self, mode: str):
        if mode == "semantic":
            if not self.embedding_model:
                raise ValueError("Embedding model is not defined.")
            return SemanticChunker(self.embedding_model)

        elif mode == "recursive":
            return RecursiveCharacterTextSplitter(
                chunk_size=1500, chunk_overlap=200, separators=["\n\n", "\n", " "]
            )
        elif mode == "strategy_0":
            return chunking_strategy_0_splitter(chunk_size=1000)
        elif mode == "strategy_1":
            print("Strategy 1")
            return chunking_strategy_1_splitter()
        elif mode == "strategy_2":
            return DualPassSplitter(
                primary_separators=["\n+[a-z]{0,2}[A-ZÉÈÀÊÔ \-'0-9\—\.]{8,}?.*\n"],
                secondary_separators=[
                    "[0-9\-\—\. ]{3,}[a-z\&'\&]*[A-ZÉÈÊÀÔ\-\.\—\°\:\; ']{10,}.*\n*",
                    "\nM\\.",
                    "\n{2,}",
                    "\n",
                ],
                chunk_size=1000,
                chunk_overlap=50,
                is_separator_regex=True,
            )
        elif mode == "strategy_3":
            return DualPassSplitter(
                primary_separators=["\n+[a-z]{0,2}[A-ZÉÈÀÊÔ \-'0-9\—\.]{8,}?.*\n"],
                secondary_separators=[
                    "[0-9\-\—\. ]{3,}[a-z\&'\&]*[A-ZÉÈÊÀÔ\-\.\—\°\:\; ']{10,}.*\n*"
                ],
                chunk_size=1000,
                chunk_overlap=50,
                is_separator_regex=True,
            )
        elif mode == "strategy_4":
            return DualPassSplitter(
                primary_separators=["\n+[a-z]{0,2}[A-ZÉÈÀÊÔ \-'0-9\—\.]{8,}?.*\n"],
                secondary_separators=[
                    "[0-9\-\—\. ]{3,}[a-z\&'\&]*[A-ZÉÈÊÀÔ\-\.\—\°\:\; ']{10,}.*\n*",
                    "\nM\\.",
                    "\n{2,}",
                    "\n",
                ],
                chunk_size=1000,
                chunk_overlap=50,
                is_separator_regex=True,
            )
        elif mode == "strategy_pp":
            return DualPassSplitter(
                primary_separators=["\n+[a-z]{0,2}[A-ZÉÈÀÊÔ \-'0-9\—\.]{8,}?.*\n"],
                secondary_separators=[
                    "[0-9\-\—\. ]{3,}[a-z\&'\&]*[A-ZÉÈÊÀÔ\-\.\—\°\:\; ']{10,}.*\n*",
                    "\nM\\.",
                ],
                chunk_size=1000,
                chunk_overlap=50,
                is_separator_regex=True,
            )
        raise ValueError(f"Unknown splitting mode: {mode}")

    def split(self, mode: str = "semantic", verbose: bool = True):
        texts, metadatas = self.load_data()
        documents = []
        splitter = self._get_splitter(mode)
        for text, meta in zip(texts, metadatas):
            try:
                docs = splitter.create_documents([text], metadatas=[meta])
                documents.extend(docs)
            except Exception as e:
                logging.error("Error during %s splitting: %s", mode, e)

        manager = DuckDBManager()
        con = duckdb.connect(manager.db_path)
        for meta in metadatas:
            source = meta.get("source", "")
            con.execute(
                "UPDATE documents SET is_already_splitted = TRUE WHERE source = ?",
                [source],
            )
        con.close()

        if verbose:
            print(f"Number of documents ({mode.capitalize()}): {len(documents)}")

        return documents

    def export_documents_to_json(
        self,
        documents: List[Document],
        filename: str,
        output_dir=path_provider.corpus_collections(),
    ):

        filename += ".json"

        output_path = Path(output_dir) / filename

        serializable_data = [
            {"content": doc.page_content, "metadata": doc.metadata} for doc in documents
        ]

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)

        print(f"✅ Exporté {len(documents)} documents dans : {output_path.resolve()}")
