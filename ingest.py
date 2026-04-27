"""CLI entry point: python ingest.py [directory]

Ingests all PDF and Markdown files in the target directory into the ChromaDB
vector store. Skips files that have already been ingested (SHA-256 duplicate
detection). Defaults to ./docs/ if no argument is given.
"""
import sys
from pathlib import Path

from src.ingestion.pipeline import ingest_directory


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/")
    if not target.exists():
        print(f"Directory not found: {target}")
        sys.exit(1)

    results = ingest_directory(target)
    print(f"\n{'File':<50} {'Status':<25} {'Chunks':>6}")
    print("-" * 85)
    for r in results:
        print(f"{r['file']:<50} {r['status']:<25} {r['chunks']:>6}")

    ingested = sum(1 for r in results if r["status"] == "ingested")
    skipped = sum(1 for r in results if r["status"] == "skipped_duplicate")
    print(f"\nDone: {ingested} ingested, {skipped} skipped (duplicate).")
