import logging
import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import spacy
from langchain_ollama.llms import OllamaLLM

# Charge les variables de .env
load_dotenv()

OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
class MetadataGenerator:
    """
    Generates metadata for a document using an LLM (for summary and theme)
    and optionally spaCy (for named entity extraction).
    """

    def __init__(self, model_name: str = OLLAMA_LLM_MODEL, spacy_model: str = "fr_core_news_sm"):
        self.chain = OllamaLLM(model=model_name)
        self.nlp = spacy.load(spacy_model)

    def generate_summary(self, text: str) -> str:
        """Use the LLM to generate a bullet-style summary."""
        print("Generating summary...")
        prompt = (
            "Donne-moi uniquement la liste des points du sommaire, sans introduction ni formules de politesse ou modalisateurs.\n\n"
            "Texte :\n" + text
        )
        try:
            return self.chain.invoke(prompt)
        except Exception as e:
            logging.error("Erreur lors de l'appel au LLM pour le sommaire : %s", e)
            return "No data"

    def generate_global_theme(self, text: str) -> str:
        """Use the LLM to extract the global topic of the document."""
        print("Generating global theme...")
        prompt = (
            "Donne-moi le contexte global du document en deux phrases maximum, sans introduction, sans formule, uniquement des informations brutes.\n\n"
            "Texte :\n" + text
        )
        try:
            return self.chain.invoke(prompt)
        except Exception as e:
            logging.error("Erreur lors de l'appel au LLM pour le thème global : %s", e)
            return "No data"

    def extract_named_entities(self, text: str) -> list:
        """Extract named entities using spaCy."""
        print("Extracting named entities...")
        doc = self.nlp(text)
        return [{"entity": ent.text, "label": ent.label_} for ent in doc.ents]
