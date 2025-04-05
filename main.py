from modules.splitting import splitter
from modules.splitting import read
from modules.embedding.store import store_documents
from langchain_ollama import OllamaEmbeddings

EMBEDDING_MODEL = "nomic-embed-text"
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

# Lecture des métadonnées depuis DuckDB
df = read.read_metadata_from_duckdb("/Users/mtis/Local/Code/GitRepos/LangAI/data/metadata.duckdb")

# Supposons que la colonne contenant le texte s'appelle "texte"
texts = df["texte"].tolist()
# Extraire toutes les autres colonnes comme métadonnées
metadata_list = df.drop(columns=["texte"]).to_dict(orient="records")

# print("Traitement en parallèle avec SemanticChunker...")
# docs_semantic = splitter.sequential_split_documents(texts, metadata_list, mode="semantic")
# print(f"Nombre de documents (Semantic): {len(docs_semantic)}")

# Vous pouvez décommenter et tester les autres modes pour comparaison
# print("Traitement en parallèle avec RecursiveCharacterTextSplitter...")
#docs_recursive = splitter.sequential_split_documents(texts, metadata_list, mode="recursive")
# print(f"Nombre de documents (Recursive): {len(docs_recursive)}")
# print("Traitement en parallèle avec CharacterTextSplitter (tiktoken)...")
docs_token = splitter.sequential_split_documents(texts, metadata_list, mode="token")
# print(f"Nombre de documents (Token-based): {len(docs_token)}")

# Stocker les documents dans ChromaDB. Assurez-vous que le namespace reflète bien la méthode utilisée.
store_documents(docs=docs_token,embedding_model=embeddings,collection_name="base_chunking")