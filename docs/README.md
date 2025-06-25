LangAI RAG System - Documentation & Architecture Overview

🚀 Project Description

This project implements a modular and extensible Retrieval-Augmented Generation (RAG) pipeline designed to analyze large corpora (e.g. parliamentary debates) using Ollama models and LangChain components.

It supports:

Metadata extraction using LLMs (summaries, themes)

Document splitting with multiple strategies

Embedding and vector storage using ChromaDB

Retrieval with advanced query augmentation (HyDE, multi-query, rewrite)

Evaluation and visualization with Streamlit

📁 Project Structure

backend/
├── RagCore/
│   ├── KnowledgeManagement/
│   │   ├── Embedding/
│   │   │   └── Embedder.py
│   │   ├── Indexing/
│   │   │   ├── duckdbManager.py
│   │   │   └── metadataGenerator.py
│   │   └── splitter.py
│   ├── Retrieving/
│   │   └── retriever.py
│   └── Utils/
│       └── pathProvider.py
frontend/
├── interface/
│   └── logic.py
├── ui/
│   └── ui_backend.py
└── web_app.py

🧠 Module Breakdown

Embedding/Embedder.py

Class ChromaEmbedderHF

Embeds and stores LangChain Document objects into a ChromaDB collection.

Uses OllamaEmbeddings model defined in .env

Handles persistence via PathProvider

Indexing/metadataGenerator.py

Class MetadataGenerator

Calls LLMs (via OllamaLLM) to generate:

Bullet-style summaries

Global themes

Uses spaCy for named entity extraction

Indexing/duckdbManager.py

Class DuckDBManager

Manages the DuckDB document database

text_file_to_duckdb method loads a file only if not already present (by source date)

Adds metadata and is_already_splitted boolean

splitter.py

Class DocumentSplitter

Loads unsplit documents from DuckDB

Supports three splitting strategies:

semantic (embedding-aware)

recursive (standard chunking)

token (Tiktoken-aware)

Marks documents as split in the DB after processing

Retrieving/retriever.py

Class RAGRetriever

Instantiates Chroma retriever with optional persistence path

Supports advanced query augmentation:

Query Rewriting

Multi-query generation

HyDE (hypothetical answer embedding)

Deduplicates and reranks documents using cosine similarity

Returns top-k scored docs

Utils/pathProvider.py

Class PathProvider

Centralized file/directory path resolver

Abstracts paths for metadata, raw data, embeddings, etc.

🎨 Frontend (Streamlit UI)

frontend/web_app.py

Runs Streamlit app

Configures sidebar and input fields

Connects UI to pipeline via logic.py

interface/logic.py

Defines run_query_pipeline() and render_query_results()

Handles query-to-retrieval-to-answer generation

Visualizes embeddings (PCA or t-SNE)

Displays retrieved documents and similarity scores

ui/ui_backend.py

Manages sidebar settings for pipeline configuration

Toggles for query rewriting, HyDE, reranking, etc.

📈 Strengths

🔁 Modular pipeline: each component (metadata, embedding, retrieval) is decoupled

🧪 Evaluation-ready: built-in Streamlit interface for scoring answers

🧠 Query augmentation: multiple strategies (rewrite, multi, HyDE)

🔍 Reranking + deduplication improves retrieval quality

🛠️ Suggestions for Improvement

Unit testing

Add tests for each component (splitter, retriever, etc.)

Logging and monitoring

Use logging consistently across all modules

Add logging level control via .env

Model abstraction

Enable switching between Ollama and HF from .env

Wrap OllamaLLM, OllamaEmbeddings in factory functions

Chunk deduplication improvements

Use hashing on content or MinHash instead of prefix matching

Metadata storage

Separate metadata and documents in DuckDB for better query performance

Caching

Cache embedding/model outputs to reduce recomputation

🗂️ .env Example

OLLAMA_LLM_MODEL=llama3.2
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
HUGGINGFACEHUB_API_TOKEN=your_token_here

✅ How to Run

Install dependencies

pip install -r requirements.txt

Initialize .env

Launch app

streamlit run frontend/web_app.py

👏 Credits

This project is a custom-built RAG architecture inspired by research best practices, designed to be modular, explainable, and extensible.