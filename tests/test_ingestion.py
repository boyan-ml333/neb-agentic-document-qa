"""Integration tests for the ingestion pipeline.

Uses a real (temporary) ChromaDB and the fixture tiny_loan.pdf.
No mocking — these tests exercise the full parse → chunk → embed → store path.
"""
import pytest
from pathlib import Path
import tempfile
import shutil

from src.store.vector_store import VectorStore
from src.ingestion.pipeline import ingest_file, ingest_directory
from src.ingestion.parsers import PdfParser, MarkdownParser


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def tmp_store(tmp_path, monkeypatch):
    """VectorStore backed by a temporary directory; cleaned up after each test."""
    chroma_dir = tmp_path / "chroma"
    monkeypatch.setattr("src.config.settings.chroma_persist_dir", chroma_dir)
    monkeypatch.setattr("src.config.settings.chroma_collection", "test_docs")
    return VectorStore()


class TestPdfParser:
    def test_parse_returns_segments(self):
        parser = PdfParser()
        segments = parser.parse(FIXTURES / "tiny_loan.pdf")
        assert len(segments) > 0
        for seg in segments:
            assert isinstance(seg.text, str)
            assert len(seg.text) > 0
            assert "page" in seg.source_metadata

    def test_page_numbers_are_sequential(self):
        parser = PdfParser()
        segments = parser.parse(FIXTURES / "tiny_loan.pdf")
        pages = [s.source_metadata["page"] for s in segments]
        assert pages == sorted(pages)


class TestMarkdownParser:
    def test_parse_md_with_headings(self, tmp_path):
        md_file = tmp_path / "sample.md"
        md_file.write_text(
            "# Title\n\nIntro paragraph.\n\n## Section A\n\nContent A.\n\n## Section B\n\nContent B.\n"
        )
        parser = MarkdownParser()
        segments = parser.parse(md_file)
        assert len(segments) >= 2
        heading_paths = [s.source_metadata.get("heading_path", "") for s in segments]
        assert any("Title" in hp or "Section A" in hp for hp in heading_paths)

    def test_parse_md_no_headings(self, tmp_path):
        md_file = tmp_path / "flat.md"
        md_file.write_text("Just some plain text with no headings at all.")
        parser = MarkdownParser()
        segments = parser.parse(md_file)
        assert len(segments) == 1
        assert "root" in segments[0].source_metadata.get("heading_path", "root")


class TestIngestFile:
    def test_ingest_pdf(self, tmp_store):
        result = ingest_file(FIXTURES / "tiny_loan.pdf", tmp_store)
        assert result["status"] == "ingested"
        assert result["chunks"] > 0

    def test_duplicate_detection(self, tmp_store):
        path = FIXTURES / "tiny_loan.pdf"
        first = ingest_file(path, tmp_store)
        second = ingest_file(path, tmp_store)
        assert first["status"] == "ingested"
        assert second["status"] == "skipped_duplicate"
        assert second["chunks"] == 0

    def test_unsupported_file_skipped(self, tmp_store, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\n1,2\n")
        result = ingest_file(csv_file, tmp_store)
        assert result["status"] == "skipped_unsupported"

    def test_search_after_ingest(self, tmp_store):
        ingest_file(FIXTURES / "tiny_loan.pdf", tmp_store)
        results = tmp_store.search("loan principal interest rate", k=3)
        assert len(results) > 0
        for doc in results:
            assert doc.page_content


class TestIngestDirectory:
    def test_ingest_directory(self, tmp_store, tmp_path):
        md_a = tmp_path / "doc_a.md"
        md_b = tmp_path / "doc_b.md"
        md_a.write_text("# Loan Terms\n\nAPR is 7.5%. Principal is $10,000.")
        md_b.write_text("# Policy\n\nPayments are due monthly.")
        shutil.copy(FIXTURES / "tiny_loan.pdf", tmp_path / "tiny_loan.pdf")

        results = ingest_directory(tmp_path, tmp_store)
        statuses = {r["file"]: r["status"] for r in results}

        assert statuses.get("doc_a.md") == "ingested"
        assert statuses.get("doc_b.md") == "ingested"
        assert statuses.get("tiny_loan.pdf") == "ingested"

    def test_list_files_after_ingest(self, tmp_store, tmp_path):
        (tmp_path / "doc.md").write_text("# Test\n\nSome content here for testing.")
        ingest_directory(tmp_path, tmp_store)
        files = tmp_store.list_files()
        assert any(f["filename"] == "doc.md" for f in files)
