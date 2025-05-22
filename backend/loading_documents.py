from pathlib import Path
import yaml
from regex import splititer

from backend.RagCore.KnowledgeManagement.Embedding.Embedder_bis import (
    ChromaEmbedder,
    get_embedder,
)
from backend.RagCore.KnowledgeManagement.Indexing.duckdbManager import DuckDBManager
from backend.RagCore.KnowledgeManagement.Indexing.documentSplitter import (
    DocumentSplitter,
)
from backend.RagCore.Utils.pathProvider import PathProvider
from backend.RagCore.Utils.fileIndexBuilder import FileIndexBuilder



def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    

path_provider = PathProvider()
path = path_provider.config_path()
params= load_config(path)
docs_splitter = DocumentSplitter(embedding_model=params["embedding_model"])
duckdbManager = DuckDBManager()


def vectorize_documents_by_model(
    docs,
    model_name: str,
    chroma_path: Path,
    collection_name: str = None,
    device: str = "cpu",
):
    """
    Vectorizes a list of documents using the specified embedding model.
    The backend is determined manually via if-statements, allowing easy future extensions.

    :param docs: List of LangChain Document objects
    :param model_name: The embedding model name (e.g. HF or Ollama)
    :param chroma_path: Path to Chroma persistent storage
    :param collection_name: Optional collection name (defaults to backend-specific name)
    :param device: 'cpu' or 'cuda'
    """
    print("Vectorizing started here")
    if "sentence-transformers" in model_name or "all-MiniLM" in model_name:
        backend = "hf"
        collection = collection_name or "hf_collection"
        print(f"⚙️  Vectorizing with HuggingFace model: {model_name}")
    elif "nomic-embed-text" in model_name or "llama" in model_name:
        backend = "ollama"
        collection = collection_name or "ollama_collection"
        print(f"⚙️  Vectorizing with Ollama model: {model_name}")
    else:
        raise ValueError(f"[❌] Unsupported or unknown model: {model_name}")

    embedder = get_embedder(backend=backend, model_name=model_name, device=device)
    chroma = ChromaEmbedder(
        collection_name=collection, embedding_backend=embedder, persist_path=chroma_path
    )
    print("Vectorbase path --------->: "+str(chroma_path))
    chroma.store_documents(docs)


"""
# Test Ollama backend
print("Test Ollama backend")
ollama_embedder = get_embedder(backend="ollama", model_name="nomic-embed-text:latest")
chroma_ollama = ChromaEmbedder(collection_name="test_ollama", embedding_backend=ollama_embedder, persist_path=Path("chroma_test_db"))
chroma_ollama.store_documents(docs)
"""


def run_loading_pipeline(
    data_source=path_provider.raw_data(),
    chroma_path="",
    chunking_strategy="",
    export_split=True,
    export_name="default",
    embedding_model="nomic-embed-text:latest",
    embedding_device_if_available="cpu",
    collection_name="default",
    advanced_metadatas=False,
):
    index_builder = FileIndexBuilder(data_source)
    chroma_dir = path_provider.data(chroma_path)
    files = index_builder.build_index()
    for key, path in files.items():
        print(f"{key} ➜ {path}")
        duckdbManager.text_file_to_duckdb(key + ".txt", metadata=advanced_metadatas)
    ## Embed and store in chromaDB
    print("Splitting started here")
    docs = docs_splitter.split(chunking_strategy)
    if export_split:
        print("exporting startingh here")
        docs_splitter.export_documents_to_json(documents=docs, filename=export_name)
    vectorize_documents_by_model(
        docs=docs,
        model_name=embedding_model,
        chroma_path=chroma_dir,
        collection_name=collection_name,
        device=embedding_device_if_available,
    )



run_loading_pipeline(**params)