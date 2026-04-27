from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.ingestion.embedder import get_embeddings
from src.config import settings


class VectorStore:
    """Thin wrapper around langchain_chroma.Chroma.

    Keeping a wrapper class gives the agent tools a stable interface even if
    the backing store changes (e.g. swap Chroma for a Snowflake VECTOR table).
    """

    def __init__(self) -> None:
        self.store = Chroma(
            collection_name=settings.chroma_collection,
            embedding_function=get_embeddings(),
            persist_directory=str(settings.chroma_persist_dir),
        )

    def upsert(self, docs: list[Document], ids: list[str]) -> None:
        """Add or update documents. Chroma treats matching IDs as upserts."""
        self.store.add_documents(documents=docs, ids=ids)

    def search(
        self,
        query: str,
        k: int = 5,
        filename: str | None = None,
    ) -> list[Document]:
        where = {"filename": filename} if filename else None
        return self.store.similarity_search(query, k=k, filter=where)

    def list_files(self) -> list[dict]:
        """Return [{filename, chunks}] for every ingested file."""
        data = self.store.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for md in data["metadatas"]:
            name = md.get("filename", "<unknown>")
            counts[name] = counts.get(name, 0) + 1
        return [{"filename": f, "chunks": n} for f, n in sorted(counts.items())]

    def file_exists(self, sha256: str) -> bool:
        """Return True if any chunk with this file-level SHA-256 is stored."""
        result = self.store.get(where={"file_sha256": sha256}, limit=1)
        return len(result["ids"]) > 0

    def get_chunks_for_file(self, filename: str) -> list[Document]:
        """Return all stored chunks for a specific filename, ordered by chunk_index."""
        result = self.store.get(
            where={"filename": filename},
            include=["documents", "metadatas"],
        )
        pairs = list(zip(result["documents"], result["metadatas"]))
        pairs.sort(key=lambda p: p[1].get("chunk_index", 0))
        return [Document(page_content=text, metadata=meta) for text, meta in pairs]
