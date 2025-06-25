from typing import List
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.prompts import PromptTemplate

from numpy.linalg import norm
from statistics import mean
import numpy as np
import logging
from backend.RagCore.Utils.configManager import ConfigManager

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("RAGRetriever")

# Cosine similarity function
def cosine_similarity(u, v):
    return np.dot(u, v) / (norm(u) * norm(v) + 1e-10)

def _deduplicate_docs(docs: list[Document]) -> list[Document]:
    log.info(f"Deduplicating {len(docs)} documents...")
    seen = set()
    unique = []
    for doc in docs:
        key = doc.page_content.strip()[:100]
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    log.info(f"{len(unique)} unique documents retained.")
    return unique

def _rerank(docs: list[tuple[Document, float]], query: str) -> list[tuple[Document, float]]:
    log.info(f"[RERANK] Reranker not implemented — passing original scores.")
    return docs

class RAGRetriever:
    def __init__(self):
        self.config = ConfigManager()

        # Load Chroma
        self.chroma = self.config.get_chroma_base()

        #Load embedder
        self.embedder = self.config.get_embedder()
        self.llm = self.config.get_llm_provider()
        self.prompts = self.config.get_retrieval_prompts()
        self.prompt_rewrite     = PromptTemplate.from_template(self.prompts["rewrite"])
        self.prompt_multi_query = PromptTemplate.from_template(self.prompts["multi_query"])
        self.prompt_hyde        = PromptTemplate.from_template(self.prompts["hyde"])

        self.eval_mode = self.config.get_eval_mode()
        self.eval_info = {}
        
        if self.eval_mode:
            log.info("Evaluation mode enabled")
        log.info("RAGRetriever initialized.")

    def _prepare_queries(self, question: str) -> list[str]:
        queries = [question]

        if self.config.get_use_query_rewriting():
            rewritten = self._rewrite_query(question)
            queries = [rewritten]
            log.info(f"→ Rewritten: {rewritten}")

        if self.config.get_use_multi_queries():
            multi = self._get_multi_queries(queries[0])
            queries += multi
            log.info(f"→ Reformulations: {multi}")

        return queries

    def _rewrite_query(self, question: str) -> str:
        tpl = self.prompt_rewrite.format(question=question)
        output = self.llm.invoke(tpl)
        return getattr(output, "content", str(output)).strip()

    def _get_multi_queries(self, question: str) -> List[str]:
        tpl = self.prompt_multi_query.format(question=question)
        output = self.llm.invoke(tpl)
        text = getattr(output, "content", str(output))
        return [q.strip("- ").strip() for q in text.splitlines() if q.strip()]

    def _get_hypothetical_answer(self, question: str) -> str:
        tpl = self.prompt_hyde.format(question=question)
        output = self.llm.invoke(tpl)
        return getattr(output, "content", str(output)).strip()

    def _search_from_query(self, question: str, top_k: int) -> list[Document]:
        queries = self._prepare_queries(question)
        all_docs: list[Document] = []

        use_hyde = self.config.get_use_hyde()
        for q in queries:
            vec_input = self._get_hypothetical_answer(q) if use_hyde else q
            embedding = self.embedder.embed_query(vec_input)

            # appelle la méthode low-level de ChromaDB
            result = self.chroma.query(
                query_embeddings=[embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            # result["documents"][0] est la liste de chaînes de caractères
            docs = [
                Document(page_content=doc_text, metadata=meta)
                for doc_text, meta in zip(
                    result["documents"][0],
                    result["metadatas"][0],
                )
            ]
            all_docs.extend(docs)
            log.info(f"→ Retrieved {len(docs)} docs for: {q}")

        # dédoublonne tes Document
        return _deduplicate_docs(all_docs)

    def retrieve(self, question: str) -> list[tuple[Document, float]]:
        top_k = self.config.get_top_k()
        docs = self._search_from_query(question, top_k)
        query_vec = self.embedder.embed_query(question)

        results = [
            (doc, cosine_similarity(self.embedder.embed_query(doc.page_content), query_vec))
            for doc in docs
        ]
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

        if self.eval_mode:
            scores = [s for _, s in sorted_results]
            self.eval_info = {
                "mean_score": mean(scores) if scores else 0.0,
                "top1_score": scores[0] if scores else None,
                "top_k": top_k
            }

        return sorted_results

    def answer(self, question: str) -> str:
        log.info(f"Question posée : {question} ")
        docs = self.retrieve(question)
        if not docs:
            return "No relevant information found in the document database."

        context = "\n\n".join(doc.page_content for doc, _ in docs)
        gen_cfg = self.config.get_generation_params()
        
        if self.eval_mode:
            self.eval_info.update({
                "context_length_chars": len(context),
                "temperature": gen_cfg["temperature"],
                "gen_model": gen_cfg["model"],
                "provider": gen_cfg["provider"],
                "embedding_model": self.embedder.model_name
            })

        prompt = PromptTemplate.from_template(
            gen_cfg["qa_prompt_template"] or
            "<s>[INST] {system_prompt}\n\nContexte :\n{context}\n\nQuestion : {question} [/INST]"
        ).format(
            system_prompt=gen_cfg["system_prompt"],
            context=context,
            question=question
        )

        response = self.llm.invoke(prompt)
        return getattr(response, "content", str(response)).strip()