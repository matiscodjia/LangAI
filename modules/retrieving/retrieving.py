from re import search
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from modules.data_managing.metadata_gen import resolve_path
from langchain_ollama import OllamaLLM, OllamaEmbeddings



def get_retriever(collection, k):
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(
        collection_name=collection,
        embedding_function=embedding_function,
        persist_directory=resolve_path("data/chroma_db")
    ).as_retriever(search_kwargs={"k" : k})

def get_qa_chain(collection, k):
        llm = OllamaLLM(model="llama3.2")
        return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=get_retriever(collection, k),
        return_source_documents=True
    )


