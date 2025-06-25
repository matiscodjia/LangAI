import logging
import os
from pathlib import Path
from typing import Any, Optional, Union, Dict

import yaml

from langchain_huggingface import HuggingFaceEmbeddings
from chromadb import PersistentClient
from dotenv import load_dotenv
from backend.RagCore.Utils.pathProvider import PathProvider


load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

class ConfigManager:
    _instance = None
    _config = None
    REQUIRED_KEYS = [
        "OPENAI_API_KEY",
        "MISTRAL_API_KEY"
    ]
    def __new__(cls, config_path: Optional[Union[str, Path]] = None):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
  
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        if self._initialized:
            return
        load_dotenv()
        self.validate_env_vars()
        from backend.RagCore.Utils.pathProvider import PathProvider
        self.path_provider = PathProvider()
        self.config_path = Path(config_path) if config_path else self.path_provider.config_path()
        self.load_config()
        self._initialized = True
    

    def validate_env_vars(self):
        """
        Vérifie que toutes les clés listées sont présentes dans les env vars.
        Lève une erreur si l'une d'elles manque, sinon logue la réussite.
        """

        missing = [k for k in self.REQUIRED_KEYS if not os.getenv(k)]
        if missing:
            log.error("Variables d'environnement manquantes : %s", missing)
            raise RuntimeError(f"Il manque les variables d'environnement suivantes : {missing}")
        else:
            log.info("Toutes les variables d'environnement requises sont chargées")


    
    def load_config(self) -> None:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

    def get_config(self) -> Dict[str, Any]:
        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        if not self._config:
            return default
        if "." in key:
            parts = key.split(".")
            value = self._config
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            return value
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        if not self._config:
            self._config = {}
        if "." in key:
            parts = key.split(".")
            config = self._config
            for part in parts[:-1]:
                if part not in config:
                    config[part] = {}
                config = config[part]
            config[parts[-1]] = value
        else:
            self._config[key] = value

    def save_config(self, path: Optional[Union[str, Path]] = None) -> None:
        save_path = Path(path) if path else self.config_path
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        if not self._config:
            self._config = {}
        def update_nested(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    update_nested(d[k], v)
                else:
                    d[k] = v
        update_nested(self._config, config_dict)

    def get_document_processing_config(self) -> Dict[str, Any]:
        return self.get("document_processing", {})

    def get_embedding_config(self) -> Dict[str, Any]:
        return self.get("embedding", {})

    def get_chromadb_config(self) -> Dict[str, Any]:
        return self.get("chromadb", {})

    def get_retrieval_config(self) -> Dict[str, Any]:
        return self.get("retrieval", {})
    def get_eval_mode(self) -> bool:
        return self.get("runtime.eval_mode", False)
    def get_generation_config(self) -> Dict[str, Any]:
        return self.get("generation", {})

    def get_visualization_config(self) -> Dict[str, Any]:
        return self.get("visualization", {})

    def get_ui_config(self) -> Dict[str, Any]:
        return self.get("ui", {})

    def get_runtime_flags(self) -> Dict[str, bool]:
        runtime = self.get("runtime", {})
        return {
            "eval_mode": runtime.get("eval_mode", False),
            "pre_retrieval": runtime.get("pre_retrieval", False),
        }

    def get_embedder(self):
        embedder = HuggingFaceEmbeddings(
        model_name=self.get("embedding.model_name"),
        model_kwargs={
            "device": self.get("embedding.device", "cpu"),
            "trust_remote_code": self.get("embedding.trust_remote_code", True)
        },
        encode_kwargs={
            "normalize_embeddings": self.get("embedding.normalize_embeddings", True)
        }


    )
        embedder.show_progress = True
        return embedder
    def get_chroma_base(self):
        chroma_path = self.get_chroma_path()
        log.info(f"Using ChromaDB path: {chroma_path}")
        client = PersistentClient(path=str(chroma_path))
        # Use configured hnsw_space or default to "cosine"
        hnsw_space = self.get_chromadb_config().get("hnsw_space", "cosine")
        return client.get_or_create_collection(
            name=self.get_collection_name(),
            metadata={"hnsw:space": hnsw_space}
        )

    def get_generation_params(self) -> Dict[str, Any]:
        gen = self.get_generation_config()
        return {
            "temperature": gen.get("temperature", 0.0),
            "model": gen.get("model", "llama3.2:latest"),
            "provider": gen.get("provider", "ollama"),
            "context_length": gen.get("context_length", 3000),
            "system_prompt": gen.get("system_prompt", ""),
            "qa_prompt_template": gen.get("qa_prompt_template", "")
        }

    def get_llm_provider(self):
        from langchain_ollama import OllamaLLM
        from langchain_openai import ChatOpenAI
        from langchain_mistralai import ChatMistralAI
        import os

        gen = self.get_generation_params()
        provider = gen["provider"]
        model = gen["model"]
        temperature = gen["temperature"]

        if provider == "openai":
            return ChatOpenAI(model=model, temperature=temperature)
        elif provider == "mistral":
            return ChatMistralAI(
                model_name=model,
                temperature=temperature,
                mistral_api_key=os.getenv("MISTRAL_API_KEY")
            )
        return OllamaLLM(model=model, temperature=temperature)






    def get_export_split(self) -> bool:
        # document_processing.export_split
        return self.get("document_processing.export_split", False)

    def get_chunking_strategy(self) -> str:
        # document_processing.chunking_strategy
        return self.get("document_processing.chunking_strategy", "strategy_1")

    def get_export_name(self) -> str:
        # document_processing.export_name
        return self.get("document_processing.export_name", "default")

    def get_raw_data_path(self) -> Path:
        # document_processing.path
        p = self.get("document_processing.path", "")
        return Path(p) if p else Path(self.path_provider.raw_data())

    def get_advanced_metadata(self) -> bool:
        # document_processing.advanced_metadata
        return self.get("document_processing.advanced_metadata", False)

    def get_collection_name(self) -> str:
        # chromadb.collection_name
        return self.get("chromadb.collection_name", str(self.path_provider.chroma()))

    def get_chroma_path(self) -> Path:
        # chromadb.path
        p = self.get("chromadb.path", "")
        return Path(p) if p else Path(self.path_provider.chroma())

    def get_use_query_rewriting(self) -> bool:
        # retrieval.use_query_rewrite
        return self.get("retrieval.use_query_rewrite", False)

    def get_use_multi_queries(self) -> bool:
        # retrieval.use_multi_query
        return self.get("retrieval.use_multi_query", False)

    def get_use_hyde(self) -> bool:
        # retrieval.use_hyde
        return self.get("retrieval.use_hyde", False)

    def get_use_rerank(self) -> bool:
        # retrieval.use_rerank
        return self.get("retrieval.use_rerank", False)

    def get_top_k(self) -> int:
        # retrieval.top_k
        return self.get("retrieval.top_k", 4)

    def get_retrieval_prompts(self) -> Dict[str, Any]:
        # retrieval.query_prompts
        return self.get("retrieval.query_prompts", {})
