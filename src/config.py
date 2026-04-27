from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Storage
    chroma_persist_dir: Path = Path("./data/chroma")
    chroma_collection: str = "documents"

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Embeddings
    embedding_provider: str = "local"            # local | openai | bedrock
    embedding_model_local: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model_openai: str = "text-embedding-3-small"
    embedding_model_bedrock: str = "amazon.titan-embed-text-v2:0"

    # LLM
    llm_provider: str = "anthropic"              # anthropic | bedrock
    llm_model: str = "claude-sonnet-4-6"
    llm_max_tokens: int = 1024
    anthropic_api_key: str | None = None         # ChatAnthropic also reads ANTHROPIC_API_KEY env var directly

    # Retrieval
    retrieval_k: int = 5

    # Agent
    max_agent_iterations: int = 8


settings = Settings()
