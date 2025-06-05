from langchain_chroma import Chroma
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.llms.base import LLM
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from typing import List, Optional
import numpy as np
from typing import List, Tuple
from backend.RagCore.Utils.pathProvider import PathProvider
import logging

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("RAGRetriever")

# Prompt templates
REWRITE_PROMPT = PromptTemplate.from_template(
    "Réécris cette question de façon plus précise pour interroger une base documentaire : {question}"
)

MULTI_QUERY_PROMPT = PromptTemplate.from_template(
    "Génère 3 reformulations différentes mais pertinentes de la question suivante : {question}"
)

HYDE_PROMPT = PromptTemplate.from_template(
    "Imagine une réponse hypothétique à cette question dans une limite stricte de 15 lignes: {question}"
)

import os
from dotenv import load_dotenv

# Charge les variables de .env
load_dotenv()

OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")


class RAGRetriever:
    def __init__(
        self,
        collection_name: str,
        persist_path: Optional[str] = None,
        gen_model=None
    ):
        self.llm: LLM = OllamaLLM(model=gen_model)
        self.embedder = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
        provider = PathProvider()
        self.chroma = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedder,
            persist_directory=persist_path or str(provider.chroma()),
        )
        self.retriever = self.chroma.as_retriever(search_kwargs={"k": 10})
        log.info(f"📚 Chroma retriever initialized with collection: {collection_name}")

    def get_qa_chain(self, k: int = 5) -> RetrievalQA:
        return RetrievalQA.from_chain_type(
            llm=self.llm, retriever=self.retriever, return_source_documents=True
        )

    def _rewrite_query(self, question: str) -> str:
        log.info(f"✍️ Rewriting question: {question}")
        chain = REWRITE_PROMPT | self.llm
        rewritten = chain.invoke(question)
        log.info(f"🔁 Rewritten: {rewritten.strip()}")
        return rewritten.strip()

    def _get_multi_queries(self, question: str) -> List[str]:
        log.info(f"🔀 Generating multi-queries for: {question}")
        chain = MULTI_QUERY_PROMPT | self.llm
        output = chain.invoke(question)
        queries = [q.strip("- ") for q in output.strip().split("\n") if q.strip()]
        log.info(f"📌 Reformulations: {queries}")
        return queries

    def _get_hypothetical_answer(self, question: str) -> str:
        log.info(f"💭 Generating hypothetical answer for: {question}")
        chain = HYDE_PROMPT | self.llm
        hypo = chain.invoke(question).strip()
        log.info(f"🧠 Hypothetical answer: {hypo}")
        return hypo

    def _deduplicate_docs(self, docs: List[Document]) -> List[Document]:
        log.info(f"🧹 Deduplicating {len(docs)} documents...")
        seen = set()
        unique = []
        for doc in docs:
            key = doc.page_content.strip()[:100]
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        log.info(f"✅ {len(unique)} unique documents retained.")
        return unique

    def _rerank(self, docs: List[Tuple[Document, float]], query: str) -> List[Tuple[Document, float]]:
        log.info(f"⚠️  [RERANK] Reranker not implemented — passing original scores.")
        return docs

    def retrieve(
        self,
        question: str,
        use_rewrite: bool = True,
        use_multi_query: bool = True,
        use_hyde: bool = True,
        use_rerank: bool = True,
        top_k: int = 5,
    ) -> List[Tuple[Document, float]]:
        log.info(f"\n🔎 Query received: {question}")
        queries = [question]
        
        if use_rewrite:
            queries[0] = self._rewrite_query(question)

        if use_multi_query:
            queries.extend(self._get_multi_queries(queries[0]))

        log.info(f"🔍 Final list of queries to search: {queries}")

        all_docs = []
        if use_hyde:
            log.info(f"🧪 Using HyDE for document retrieval.")
            for q in queries:
                hypo = self._get_hypothetical_answer(q)
                hypo_embedding = self.embedder.embed_query(hypo)
                hyde_docs = self.chroma.similarity_search_by_vector(hypo_embedding, k=top_k)
                log.info(f"🔹 Retrieved {len(hyde_docs)} docs for HyDE on: {q}")
                all_docs.extend(hyde_docs)
        else:
            log.info(f"📥 Retrieving documents using direct similarity search.")
            for q in queries:
                docs = self.retriever.invoke(q)
                log.info(f"🔹 Retrieved {len(docs)} docs for query: {q}")
                all_docs.extend(docs)

        cleaned_docs = self._deduplicate_docs(all_docs)

        log.info(f"📐 Computing similarity scores for {len(cleaned_docs)} documents...")
        query_vec = self.embedder.embed_query(question)
        scored = [
            (doc, np.dot(self.embedder.embed_query(doc.page_content), query_vec))
            for doc in cleaned_docs
        ]

        if use_rerank:
            log.info(f"🔄 Passing through reranker (currently identity function).")
            reranked = self._rerank(scored, question)
            return reranked[:top_k]

        log.info(f"📤 Returning top-{top_k} scored documents without reranking.")
        return scored[:top_k]