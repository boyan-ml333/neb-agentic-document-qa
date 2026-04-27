from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.ingestion.parsers import ParsedSegment


def chunk_segments(
    segments: list[ParsedSegment],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    """Split ParsedSegments into smaller chunks.

    Returns a list of dicts: {text, metadata} where metadata merges the
    segment's source_metadata with a sequential chunk_index across all segments.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []
    chunk_index = 0

    for segment in segments:
        sub_texts = splitter.split_text(segment.text)
        for sub_text in sub_texts:
            if not sub_text.strip():
                continue
            metadata = dict(segment.source_metadata)
            metadata["chunk_index"] = chunk_index
            chunks.append({"text": sub_text, "metadata": metadata})
            chunk_index += 1

    return chunks
