"""Agent tool definitions.

Each function is decorated with @tool so LangGraph's ToolNode can
introspect the JSON schema from type hints and docstrings — no glue code.

Call build_tools(store) once at startup to bind the VectorStore instance
and receive the list to pass into create_react_agent.
"""
import json
import re

from langchain_core.tools import tool

from src.store.vector_store import VectorStore

_store: VectorStore | None = None


def build_tools(store: VectorStore) -> list:
    """Bind the module-level store and return all tools for LangGraph's ToolNode."""
    global _store
    _store = store
    return [
        search_documents,
        list_documents,
        summarize_document,
        extract_loan_terms,
        compute_loan_payment,
    ]


@tool
def search_documents(query: str, k: int = 5, filename: str | None = None) -> str:
    """Semantic search across all ingested documents.

    Returns up to k chunks ranked by relevance, each with its source filename,
    page number (PDFs) or heading path (Markdown), and the chunk text.
    Use the optional filename argument to restrict results to a single file.
    Always call this tool before answering factual questions about documents.
    """
    docs = _store.search(query, k=k, filename=filename)
    if not docs:
        return "No matching chunks found."
    parts = []
    for d in docs:
        meta = d.metadata
        location = (
            f"p.{meta['page']}" if meta.get("page")
            else meta.get("heading_path", "")
        )
        parts.append(
            f"[{meta.get('filename', '?')} {location}]\n{d.page_content}"
        )
    return "\n\n---\n\n".join(parts)


@tool
def list_documents() -> str:
    """List every ingested document with its chunk count.

    Use this when the user asks 'what documents do you have?' or
    'what do you know about?' before attempting any search.
    Returns a JSON array of {filename, chunks} objects.
    """
    files = _store.list_files()
    return json.dumps(files, indent=2)


@tool
def summarize_document(filename: str) -> str:
    """Return the full text of a specific ingested document for summarization.

    Use this when the user asks for an overview or summary of a single file.
    Pass the exact filename as shown by list_documents.
    Returns all chunks concatenated in order; the model should then synthesize.
    """
    docs = _store.get_chunks_for_file(filename)
    if not docs:
        return (
            f"No file named '{filename}' found. "
            "Use list_documents to see available files."
        )
    return "\n\n".join(d.page_content for d in docs)


@tool
def extract_loan_terms(filename: str) -> str:
    """Extract structured loan terms from a loan document.

    Returns a JSON object with keys: principal, apr_pct, term_months,
    monthly_payment, origination_fee_pct. Values are numbers or null if
    not found in the document.

    Use this before compute_loan_payment when the user asks about loan
    calculations, or when comparing terms across multiple loan documents.
    """
    docs = _store.get_chunks_for_file(filename)
    if not docs:
        return json.dumps({"error": f"File '{filename}' not found."})

    text = " ".join(d.page_content for d in docs)

    def find_first(patterns: list[str]) -> float | None:
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1).replace(",", ""))
                except (ValueError, IndexError):
                    continue
        return None

    principal = find_first([
        r"loan\s+amount[:\s]+\$?([\d,]+)",
        r"loan\s+amount\s+is\s+\$?([\d,]+)",
        r"principal[:\s]+\$?([\d,]+)",
        r"amount\s+financed[:\s]+\$?([\d,]+)",
        r"\$\s*([\d,]+(?:\.\d+)?)\s*(?:personal\s+)?loan",
    ])

    apr = find_first([
        r"annual\s+percentage\s+rate[:\s]+([\d.]+)\s*%",
        r"\bAPR\s+is\s+([\d.]+)\s*%",
        r"\bAPR[:\s]+([\d.]+)\s*%",
        r"interest\s+rate[:\s]+([\d.]+)\s*%",
        r"rate[:\s]+([\d.]+)\s*%\s*(?:per\s+annum|annual|APR)",
        r"the\s+APR\s+is\s+([\d.]+)\s*%",
    ])

    term = find_first([
        r"term[:\s]+(\d+)\s*months",
        r"(\d+)[- ]month\s+(?:loan|term)",
        r"repayment\s+period[:\s]+(\d+)\s*months",
        r"(\d+)\s*monthly\s+(?:installments|payments)",
        r"over\s+(\d+)\s*months",
    ])

    monthly_payment = find_first([
        r"monthly\s+(?:payment|installment)[:\s]+\$?([\d,]+(?:\.\d+)?)",
        r"(?:payment|installment)\s+of\s+\$?([\d,]+(?:\.\d+)?)\s*per\s+month",
    ])

    origination_fee = find_first([
        r"origination\s+fee[:\s]+([\d.]+)\s*%",
        r"origination[:\s]+([\d.]+)\s*%",
    ])

    result = {
        "filename": filename,
        "principal": principal,
        "apr_pct": apr,
        "term_months": int(term) if term else None,
        "monthly_payment": monthly_payment,
        "origination_fee_pct": origination_fee,
    }
    return json.dumps(result, indent=2)


@tool
def compute_loan_payment(
    principal: float,
    annual_rate_pct: float,
    term_months: int,
) -> str:
    """Compute the monthly payment for a fixed-rate amortizing loan.

    Uses the standard formula: M = P * r(1+r)^n / ((1+r)^n - 1)
    where r = monthly rate (APR / 12 / 100) and n = term in months.

    Always use this tool for loan payment calculations — never compute by hand.
    Deterministic arithmetic in a tool is auditable; LLM arithmetic is not.
    """
    r = (annual_rate_pct / 100) / 12
    n = term_months
    if r == 0:
        monthly = principal / n
    else:
        monthly = principal * r * (1 + r) ** n / ((1 + r) ** n - 1)

    total_paid = monthly * n
    total_interest = total_paid - principal

    return (
        f"Monthly payment: ${monthly:,.2f}\n"
        f"  Principal: ${principal:,.2f}\n"
        f"  APR: {annual_rate_pct}%\n"
        f"  Term: {term_months} months\n"
        f"  Total paid: ${total_paid:,.2f}\n"
        f"  Total interest: ${total_interest:,.2f}"
    )
