"""Tests for the five agent tools.

Uses a real (temporary) VectorStore populated with fixture data so tool
behavior can be verified without mocking the retrieval layer.
"""
import json
import pytest
from pathlib import Path

from src.store.vector_store import VectorStore
from src.ingestion.pipeline import ingest_file, ingest_directory
import src.agent.tools as tools_module
from src.agent.tools import build_tools

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def populated_store(tmp_path_factory):
    """Module-scoped store populated once with fixture + inline MD docs."""
    tmp = tmp_path_factory.mktemp("store")

    import src.config as cfg
    cfg.settings.chroma_persist_dir = tmp / "chroma"
    cfg.settings.chroma_collection = "test_tools"

    store = VectorStore()

    # Ingest tiny fixture PDF
    ingest_file(FIXTURES / "tiny_loan.pdf", store)

    # Inline markdown docs for tool tests
    faq = tmp / "faq.md"
    faq.write_text(
        "# FAQ\n\n## Rates\n\nThe APR is 8.5%. The loan amount is $15,000 "
        "over 36 months. Origination fee is 1.5%.\n\n"
        "## Prepayment\n\nNo prepayment penalties apply.\n"
    )
    policy = tmp / "policy.md"
    policy.write_text(
        "# Compliance\n\n## Subprime\n\nBorrowers below 620 FICO require "
        "second-level approval and 43% DTI cap.\n"
    )
    ingest_directory(tmp, store)

    # Bind tools to this store
    build_tools(store)
    return store


class TestSearchDocuments:
    def test_returns_results(self, populated_store):
        build_tools(populated_store)
        result = tools_module.search_documents.invoke({"query": "loan principal interest rate"})
        assert isinstance(result, str)
        assert len(result) > 0
        assert "No matching chunks found" not in result

    def test_no_results_on_empty_store(self, tmp_path, monkeypatch):
        """search_documents returns the sentinel string when the store is empty."""
        monkeypatch.setattr("src.config.settings.chroma_persist_dir", tmp_path / "chroma")
        monkeypatch.setattr("src.config.settings.chroma_collection", "empty_test")
        empty_store = VectorStore()
        build_tools(empty_store)
        result = tools_module.search_documents.invoke(
            {"query": "loan interest rate APR principal"}
        )
        assert "No matching chunks found" in result

    def test_filename_filter(self, populated_store):
        build_tools(populated_store)
        result = tools_module.search_documents.invoke(
            {"query": "prepayment", "filename": "faq.md"}
        )
        assert "faq.md" in result


class TestListDocuments:
    def test_returns_json_list(self, populated_store):
        build_tools(populated_store)
        result = tools_module.list_documents.invoke({})
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) > 0
        for entry in data:
            assert "filename" in entry
            assert "chunks" in entry
            assert entry["chunks"] > 0


class TestSummarizeDocument:
    def test_returns_content(self, populated_store):
        build_tools(populated_store)
        result = tools_module.summarize_document.invoke({"filename": "faq.md"})
        assert "APR" in result or "prepayment" in result.lower()

    def test_missing_file_message(self, populated_store):
        build_tools(populated_store)
        result = tools_module.summarize_document.invoke({"filename": "nonexistent.pdf"})
        assert "nonexistent.pdf" in result and "found" in result.lower()


class TestExtractLoanTerms:
    def test_extracts_from_faq(self, populated_store):
        build_tools(populated_store)
        result = tools_module.extract_loan_terms.invoke({"filename": "faq.md"})
        data = json.loads(result)
        assert data.get("apr_pct") == pytest.approx(8.5, abs=0.1)
        assert data.get("principal") == pytest.approx(15000, abs=1)
        assert data.get("term_months") == 36

    def test_missing_file_returns_error(self, populated_store):
        build_tools(populated_store)
        result = tools_module.extract_loan_terms.invoke({"filename": "ghost.pdf"})
        data = json.loads(result)
        assert "error" in data


class TestComputeLoanPayment:
    def test_standard_calculation(self):
        result = tools_module.compute_loan_payment.invoke(
            {"principal": 10000.0, "annual_rate_pct": 6.0, "term_months": 12}
        )
        # Correct monthly payment ≈ $860.66
        assert "$860" in result or "$861" in result

    def test_zero_rate_loan(self):
        result = tools_module.compute_loan_payment.invoke(
            {"principal": 12000.0, "annual_rate_pct": 0.0, "term_months": 12}
        )
        assert "$1,000.00" in result

    def test_output_includes_summary_fields(self):
        result = tools_module.compute_loan_payment.invoke(
            {"principal": 20000.0, "annual_rate_pct": 7.5, "term_months": 48}
        )
        assert "Monthly payment" in result
        assert "Total paid" in result
        assert "Total interest" in result
