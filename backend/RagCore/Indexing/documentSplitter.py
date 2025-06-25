import json
from pathlib import Path
import duckdb
import logging
from backend.RagCore.Indexing.databaseManager import DuckDBManager
from langchain_experimental.text_splitter import SemanticChunker

from backend.RagCore.Utils.configManager import ConfigManager
from backend.RagCore.Utils.pathProvider import PathProvider
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
path_provider = PathProvider()

log = logging.getLogger("DocumentSplitter")
logging.basicConfig(level=logging.INFO, format="%(message)s")
# Custom chunking strategies
def chunking_strategy_0_splitter(chunk_size):
    return RecursiveCharacterTextSplitter(
        separators=["\nM\\.", "\n{2,}", "\n"],
        chunk_size=chunk_size,
        chunk_overlap=1,
        length_function=len,
        is_separator_regex=True,
    )


def chunking_strategy_1_splitter(chunk_size):
    return RecursiveCharacterTextSplitter(
        separators=['\n+[a-z]{0,2}[A-ZÉÈÀÊÔ \-\'0-9\—\.]{8,}?.*\n',"[0-9\-\—\. ]{3,}[a-z\&\'\&]*[A-ZÉÈÊÀÔ\-\.\—\°\:\; ']{10,}.*\n*"],
        chunk_size=chunk_size,
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


def load_data():
    manager = DuckDBManager()
    con = duckdb.connect(manager.db_path)
    df = con.execute(
        "SELECT * FROM documents"
    ).fetchdf()
    con.close()
    texts = df["texte"].tolist()
    metadata_list = df.drop(columns=["texte"]).to_dict(orient="records")
    return texts, metadata_list


def export_documents_to_json(
        documents: List[Document],
        output_dir=path_provider.corpus_collections(),
):
    config_manager = ConfigManager()
    filename = config_manager.get_export_name()
    filename += ".json"
    output_path = Path(output_dir) / filename

    serializable_data = [
        {"content": doc.page_content, "metadata": doc.metadata} for doc in documents
    ]

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(serializable_data, f, ensure_ascii=False, indent=2)

    log.info(f"Exported {len(documents)} documents to: {output_path.resolve()}")


class DocumentSplitter:
    def __init__(self,bi_encoder=None):
        self.embedding_model = bi_encoder

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
            return chunking_strategy_1_splitter(chunk_size=1000)
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

    def split(self, verbose: bool = True):
        config_manager = ConfigManager()
        mode = config_manager.get_chunking_strategy()
        texts, metadata = load_data()
        documents = []
        splitter = self._get_splitter(mode)

        for text, meta in zip(texts, metadata):
            try:
                log.info(f"Original text: {len(text.split())} words")
                docs = splitter.create_documents([text], metadatas=[meta])
                for i, doc in enumerate(docs):
                    content = doc.page_content if hasattr(doc, "page_content") else doc["page_content"]
                    num_words = len(content.split())
                    log.info(f"  Split {i + 1}: {num_words} words")
                    documents.append(doc)
            except Exception as e:
                log.error(f"Error during {mode} splitting: {e}")

        if verbose:
            log.info(f"Total documents generated with strategy '{mode}': {len(documents)}")

        return documents

