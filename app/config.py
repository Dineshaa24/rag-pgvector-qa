"""Settings read from environment variables, with defaults that work for local development."""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/rag")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    # Any OpenAI-compatible endpoint works: OpenAI, Azure OpenAI, Ollama, vLLM.
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_base_url: str | None = os.getenv("LLM_BASE_URL")
    llm_api_key: str | None = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key or self.llm_base_url)


settings = Settings()
