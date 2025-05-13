from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from backend.RagCore.KnowledgeManagement.Indexing.metadataGenerator import resolve_path
from langchain_ollama import OllamaLLM, OllamaEmbeddings
import numpy as np
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.llms.base import LLM
from typing import List
import numpy as np

def get_retriever(collection, k, metadata_filter=None):
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    chroma = Chroma(
        collection_name=collection,
        embedding_function=embedding_function,
        persist_directory=resolve_path("data/chroma_db")
    )

    search_kwargs = {"k": k}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    return chroma.as_retriever(search_kwargs=search_kwargs)

def get_qa_chain(collection, k):
        llm = OllamaLLM(model="llama3.2")
        return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=get_retriever(collection, k),
        return_source_documents=True
    )

##Prompts pour QUERY REWRITING - QUERY TRANSFORMING - HyDE cherceh ldes question plus pertinentes
REWRITE_PROMPT = PromptTemplate.from_template(
    "Réécris cette question de façon plus précise pour interroger une base documentaire : {question}"
)

MULTI_QUERY_PROMPT = PromptTemplate.from_template(
    "Génère 3 reformulations différentes mais pertinentes de la question suivante : {question}"
)

HYDE_PROMPT = PromptTemplate.from_template(
    "Imagine une réponse hypothétique à cette question : {question}"
)


def rewrite_query(llm: LLM, question: str) -> str:
    chain = LLMChain(llm=llm, prompt=REWRITE_PROMPT)
    return chain.invoke(question)["text"]

## Pour l'instant ici c'est une query composée de plusieurs queries l'idée est de faire circuler dans des flux parallèles les queries obtenues
def get_multi_queries(llm: LLM, question: str) -> List[str]:
    chain = LLMChain(llm=llm, prompt=MULTI_QUERY_PROMPT)
    output = chain.invoke(question)["text"]
    return [q.strip("- ") for q in output.strip().split("\n") if q.strip()]

## Ici aussi besoin de générer de veritables documents et non une simple réponse
def get_hypothetical_answer(llm: LLM, question: str) -> str:
    chain = LLMChain(llm=llm, prompt=HYDE_PROMPT)
    return chain.invoke(question)["text"]


## Ici c'est juste mieux que rine à voir sur la manière de comparer les documents (sémantique c'est mieux)
def deduplicate_docs_naive(documents: List[Document]) -> List[Document]:
    seen = set()
    unique_docs = []
    for doc in documents:
        key = doc.page_content.strip()[:100] 
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)
    return unique_docs

def basic_rerank(docs: List[Document], query: str, embedder) -> List[Document]:
    """
    Classe un ensemble de documents selon leur similarité  avec la requête, via embeddings.
    """
    query_vec = embedder.embed_query(query)
    scores = [np.dot(embedder.embed_query(doc.page_content), query_vec) for doc in docs]
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs]

def enhanced_retrieve(
    llm,
    retriever,
    question: str,
    embedder,
    use_query_rewrite: bool = True,
    use_multi_query: bool = True,
    use_hyde: bool = True,
    top_k: int = 10,
) -> List[Document]:
    queries = [question]

    if use_query_rewrite:
        rewritten = rewrite_query(llm, question)
        queries.append(rewritten)
    else:
        rewritten = question

    if use_multi_query:
        multi_queries = get_multi_queries(llm, rewritten)
        queries.extend(multi_queries)
    all_docs = []
    for q in queries:
        docs = retriever.invoke(q)
        all_docs.extend(docs)

    if use_hyde:
        hypo = get_hypothetical_answer(llm, question)
        hypo_embedding = embedder.embed_query(hypo)
        hyde_docs = retriever.vectorstore.similarity_search_by_vector(hypo_embedding, k=top_k)
        all_docs.extend(hyde_docs)

    fused = deduplicate_docs_naive(all_docs)
    reranked = basic_rerank(fused, question, embedder)
    return reranked[:top_k]
