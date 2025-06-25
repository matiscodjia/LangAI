import logging

from langchain_core.messages import BaseMessage
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from backend.RagCore.Utils.configManager import ConfigManager


def extract_named_entities(text: str) -> list:
    logging.info("NER non implémenté pour l'instant.")
    return []


class MetadataGenerator:
    """
    Generates metadata (summary, theme, etc.) using a configurable LLM backend.
    """

    def __init__(self, model_name: str = None):
        config = ConfigManager()
        self.model_name = model_name or config.get("generation.model", "llama3.2:latest")
        self.model_name = self.model_name.lower()

        if "ollama" in self.model_name or ":" in self.model_name:
            self.llm = OllamaLLM(model=self.model_name.split(":")[-1])
        elif "openai" in self.model_name:
            self.llm = ChatOpenAI(model=self.model_name)
        elif "mistral" in self.model_name:
            self.llm = ChatMistralAI(model=self.model_name)
        else:
            raise ValueError(f"Unsupported LLM backend: {self.model_name}")

        logging.info(f"MetadataGenerator initialized with model: {self.model_name}")

    def generate_summary(self, text: str) -> BaseMessage | str:
        prompt = (
            "Donne-moi uniquement la liste des points du sommaire, sans introduction ni formules de politesse ou modalisateurs.\n\n"
            f"Texte :\n{text}"
        )
        try:
            return self.llm.invoke(prompt)
        except Exception as e:
            logging.error("Erreur LLM pour le sommaire : %s", e)
            return "No data"

    def generate_global_theme(self, text: str) -> BaseMessage | str:
        prompt = (
            "Donne-moi le contexte global du document en deux phrases maximum, sans introduction, sans formule, uniquement des informations brutes.\n\n"
            f"Texte :\n{text}"
        )
        try:
            return self.llm.invoke(prompt)
        except Exception as e:
            logging.error("Erreur LLM pour le thème global : %s", e)
            return "No data"

