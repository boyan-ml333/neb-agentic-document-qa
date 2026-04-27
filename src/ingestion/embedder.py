from langchain_core.embeddings import Embeddings
from src.config import settings


def get_embeddings() -> Embeddings:
    """Factory returning a LangChain Embeddings instance based on settings.

    Returning the LangChain Embeddings interface (not a custom protocol) lets
    us pass this directly into langchain_chroma.Chroma — no glue code needed.

    Local MiniLM is the banking-grade default: no PII leaves the machine.
    """
    if settings.embedding_provider == "local":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model_local,
            encode_kwargs={"normalize_embeddings": True},
        )
    elif settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=settings.embedding_model_openai)
    elif settings.embedding_provider == "bedrock":
        from langchain_aws import BedrockEmbeddings
        return BedrockEmbeddings(model_id=settings.embedding_model_bedrock)
    raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")
