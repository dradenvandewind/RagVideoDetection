"""Configuration centralisée de l'application.

Remplace les appels os.getenv() dispersés dans main.py par un objet
unique, typé, et facile à surcharger dans les tests (Settings(...)).
"""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Chroma / index
    chroma_path: str = "/app/chroma_db"
    chroma_collection: str = "video_rag"

    # LLM (Ollama)
    ollama_model: str = "llama3.2:1b"
    ollama_base_url: str = "http://ollama:11434"
    llm_temperature: float = 0.1
    llm_request_timeout: float = 300.0

    # Embeddings / chunking
    embed_model: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 512
    chunk_overlap: int = 64

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Settings mis en cache : lu une seule fois, réutilisé partout via Depends()."""
    return Settings()