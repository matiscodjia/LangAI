import types
from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from backend.RagCore.Embedding.Embedder import ChromaEmbedder
from backend.RagCore.Indexing.databaseManager import DuckDBManager
from backend.RagCore.Indexing.documentSplitter import DocumentSplitter, export_documents_to_json
from backend.RagCore.Utils.pathProvider import PathProvider
from backend.RagCore.Utils.fileIndexBuilder import FileIndexBuilder
from backend.RagCore.Utils.configManager import ConfigManager

import referencing
from referencing import Registry, Resource
import json
import urllib.request

# ─── PATCH pour jsonschema draft-03 ─────────────────────────────────────────────


# Initialize configuration manager
config_manager = ConfigManager()
docs_splitter = DocumentSplitter()
duckdbManager = DuckDBManager()

def run_loading_pipeline():
    """
    Run the document loading and embedding pipeline.
    """
    export_split = config_manager.get_export_split()
    # Get path provider
    path_provider = PathProvider()

    # Build index and process files
    index_builder = FileIndexBuilder()
    files = index_builder.build_index()

    for key, file_path in files.items():
        print(f"{file_path} ➜ {key}")
        duckdbManager.text_file_to_duckdb(str(path_provider.raw_data(file_path)))

    # Split documents
    docs = docs_splitter.split()

    # Export split documents if requested
    if export_split:
        export_documents_to_json(documents=docs)

    # Vectorize documents
    chroma_embedder = ChromaEmbedder()
    chroma_embedder.store_documents(docs)

# Run the pipeline with the configuration
run_loading_pipeline()