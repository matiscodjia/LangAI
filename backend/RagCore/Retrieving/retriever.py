from langchain_chroma import Chroma
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.llms.base import LLM
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from typing import List, Optional
import numpy as np

from backend.RagCore.Utils.pathProvider import PathProvider

# Prompt templates
REWRITE_PROMPT = PromptTemplate.from_template(
    "Réécris cette question de façon plus précise pour interroger une base documentaire : {question}"
)

MULTI_QUERY_PROMPT = PromptTemplate.from_template(
    "Génère 3 reformulations différentes mais pertinentes de la question suivante : {question}"
)

HYDE_PROMPT = PromptTemplate.from_template(
    "Imagine une réponse hypothétique à cette question : {question}"
)

import os
from dotenv import load_dotenv

# Charge les variables de .env
load_dotenv()

OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")


class RAGRetriever:
    def __init__(
        self,
        collection_name: str,
        persist_path: Optional[str] = None,
    ):
        self.llm: LLM = OllamaLLM(model=OLLAMA_LLM_MODEL)
        self.embedder = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
        provider = PathProvider()
        self.chroma = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedder,
            persist_directory=persist_path or str(provider.chroma()),
        )
        self.retriever = self.chroma.as_retriever(search_kwargs={"k": 10})

    def get_qa_chain(self, k: int = 5) -> RetrievalQA:
        return RetrievalQA.from_chain_type(
            llm=self.llm, retriever=self.retriever, return_source_documents=True
        )

    def _rewrite_query(self, question: str) -> str:
        chain = REWRITE_PROMPT | self.llm
        return chain.invoke(question)

    def _get_multi_queries(self, question: str) -> List[str]:
        chain = MULTI_QUERY_PROMPT | self.llm
        output = chain.invoke(question)
        return [q.strip("- ") for q in output.strip().split("\n") if q.strip()]

    def _get_hypothetical_answer(self, question: str) -> str:
        chain = HYDE_PROMPT | self.llm
        return chain.invoke(question)

    def _deduplicate_docs(self, docs: List[Document]) -> List[Document]:
        seen = set()
        unique = []
        for doc in docs:
            key = doc.page_content.strip()[:100]
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        return unique

    def _rerank(self, docs: List[Document], query: str) -> List[Document]:
        query_vec = self.embedder.embed_query(query)
        scores = [
            np.dot(self.embedder.embed_query(doc.page_content), query_vec)
            for doc in docs
        ]
        scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return scored  # [doc for doc, _ in scored]

    def retrieve(
        self,
        question: str,
        use_rewrite: bool = True,
        use_multi_query: bool = True,
        use_hyde: bool = True,
        top_k: int = 10,
    ) -> List[Document]:
        queries = [question]

        if use_rewrite:
            rewritten = self._rewrite_query(question)
            queries.append(rewritten)
        else:
            rewritten = question

        if use_multi_query:
            queries.extend(self._get_multi_queries(rewritten))

        all_docs = []
        for q in queries:
            docs = self.retriever.invoke(q)
            all_docs.extend(docs)

        if use_hyde:
            hypo = self._get_hypothetical_answer(question)
            hypo_embedding = self.embedder.embed_query(hypo)
            hyde_docs = self.chroma.similarity_search_by_vector(hypo_embedding, k=top_k)
            all_docs.extend(hyde_docs)

        fused = self._deduplicate_docs(all_docs)
        scored_docs = self._rerank(fused, question)
        return scored_docs[:top_k]  # List[(Document, float)]
