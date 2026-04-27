import hashlib
from pathlib import Path

from langchain_core.documents import Document

from src.ingestion.chunker import chunk_segments
from src.ingestion.parsers import PdfParser, MarkdownParser, ParsedSegment
from src.store.vector_store import VectorStore
from src.config import settings
from src.logging_setup import logger

SUPPORTED_SUFFIXES = {".pdf", ".md"}
EXCLUDED_FILENAMES = {"readme.md"}  # documentation files, not corpus documents


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _parse(path: Path) -> list[ParsedSegment]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PdfParser().parse(path)
    elif suffix == ".md":
        return MarkdownParser().parse(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def ingest_file(path: Path, store: VectorStore) -> dict:
    """Parse, chunk, embed, and upsert a single file.

    Returns {"status": "ingested"|"skipped_duplicate"|"skipped_unsupported", "chunks": N}.
    SHA-256 check at the file level short-circuits before any parsing.
    Chunk IDs are deterministic so upsert is always idempotent as a second layer.
    """
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        return {"file": path.name, "status": "skipped_unsupported", "chunks": 0}

    if path.name.lower() in EXCLUDED_FILENAMES:
        return {"file": path.name, "status": "skipped_excluded", "chunks": 0}

    sha = file_sha256(path)
    if store.file_exists(sha):
        logger.info("duplicate_skip", file=path.name)
        return {"file": path.name, "status": "skipped_duplicate", "chunks": 0}

    logger.info("ingesting", file=path.name)
    segments = _parse(path)
    raw_chunks = chunk_segments(segments, settings.chunk_size, settings.chunk_overlap)

    docs: list[Document] = []
    ids: list[str] = []

    sha_prefix = hashlib.sha256(path.name.encode()).hexdigest()[:16]

    for chunk in raw_chunks:
        chunk_index = chunk["metadata"]["chunk_index"]
        chunk_id = f"{sha_prefix}:{chunk_index}"

        metadata = {
            "chunk_id": chunk_id,
            "filename": path.name,
            "file_sha256": sha,
            "file_type": path.suffix.lstrip(".").lower(),
            "chunk_index": chunk_index,
            **chunk["metadata"],
        }
        # Normalise: ensure page/heading_path keys are always present
        metadata.setdefault("page", None)
        metadata.setdefault("heading_path", None)

        # Chroma rejects None values in metadata; replace with sentinel strings
        metadata = {
            k: (v if v is not None else "") for k, v in metadata.items()
        }

        docs.append(Document(page_content=chunk["text"], metadata=metadata))
        ids.append(chunk_id)

    store.upsert(docs, ids)
    logger.info("ingested", file=path.name, chunks=len(docs))
    return {"file": path.name, "status": "ingested", "chunks": len(docs)}


def ingest_directory(dir_path: Path, store: VectorStore | None = None) -> list[dict]:
    """Ingest all supported files in a directory (non-recursive)."""
    if store is None:
        store = VectorStore()

    results = []
    files = sorted(
        p for p in dir_path.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )

    if not files:
        logger.warning("no_supported_files", directory=str(dir_path))

    for file_path in files:
        result = ingest_file(file_path, store)
        results.append(result)

    return results
