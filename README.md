# Agentic Document Q&A

A document ingestion pipeline and agentic chatbot for question-answering over PDF and Markdown
files.

---

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │    Streamlit UI (src/ui/app.py)     │
                    │  - file uploader                    │
                    │  - chat input + history             │
                    │  - tool-call trace expander         │
                    └────────────┬────────────────────────┘
                                 │
                  ┌──────────────┴──────────────┐
                  ▼                              ▼
        ┌──────────────────┐          ┌──────────────────────┐
        │  Ingestion Pipe  │          │   Agent Loop         │
        │  src/ingestion/  │          │   src/agent/agent.py │
        └────────┬─────────┘          └──────────┬───────────┘
                 │                               │
        Parser → Chunker → Embedder              │ tool calls
                 │                               │
                 ▼                               ▼
              ┌──────────────────────────────────────┐
              │   ChromaDB (persisted to ./data/)    │
              │   metadata: filename, page,          │
              │             chunk_id, sha256         │
              └──────────────────────────────────────┘
                                 ▲
                                 │ search_documents, list_documents,
                                 │ summarize_document, extract_loan_terms,
                                 │ compute_loan_payment
                                 │
                            (Tool registry)
```

**LangGraph agent state machine** (4 nodes):

```
__start__ → agent → (tool_calls?) → tools → agent → … → __end__
```

The conditional edge after `agent`: if the LLM response contains `tool_calls`, route to
`ToolNode`; otherwise route to `END`. `MemorySaver` checkpointer persists state per
`thread_id` — same thread = conversational memory, different thread = clean slate.

---

## Quick start

```bash
# 1. Create and activate venv
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Set ANTHROPIC_API_KEY=sk-ant-... in .env

# 4. Ingest documents
python ingest.py docs/

# 5. Launch the UI
streamlit run src/ui/app.py
```

The UI lets you upload additional PDF/MD files from the browser; they are ingested
on-the-fly and immediately available for questioning.

---

## Tech choices

| Layer | Choice | Why |
|-------|--------|-----|
| PDF parsing | **PyMuPDF** (`fitz`) | Fastest pure-Python PDF lib; exposes per-page text + bounding boxes |
| MD parsing | **markdown-it-py** | Spec-compliant AST; splits on headings, preserving `heading_path` metadata |
| Chunking | `RecursiveCharacterTextSplitter` | Hierarchical splits (paragraph → sentence → word); never cuts mid-sentence |
| Embeddings | **`all-MiniLM-L6-v2`** (local) | 80MB, 384 dims, no PII egress — banking-grade default; pluggable to OpenAI/Bedrock via env var |
| Vector store | **ChromaDB** via `langchain-chroma` | Disk-persisted, metadata filtering, zero infra; same LangChain interface as Snowflake/Bedrock swap |
| Agent | **LangGraph** `create_react_agent` | Explicit state machine (auditable, visualisable); `MemorySaver` for memory; ports to Bedrock by swapping `ChatAnthropic` → `ChatBedrockConverse` |
| LLM | **Claude `claude-sonnet-4-6`** | Strong tool-use, 200K context; same code path moves to Bedrock |
| UI | **Streamlit** | Built-in file uploader and chat widgets; single-command deploy |
| Config | `pydantic-settings` | Type-checked env vars, fails loud on missing keys |
| Logging | `structlog` | JSON-structured → shippable to CloudWatch/Datadog/Snowflake later |

**Duplicate detection:** two-layer. File-level SHA-256 check before parsing (O(1) lookup).
Chunk IDs are `sha256(filename)[:16]:<chunk_index>` — deterministic, so upsert is
idempotent even if the SHA check is bypassed.

---

## Sample Q&A pairs

The following traces were run against the populated knowledge base. Each shows the
question, the tool calls the agent made, and the final answer.

### Q1 — Discovery

**Q:** *"What documents do you have access to?"*

**Tools invoked:** `list_documents`

**Tool output (abridged):**
```json
[
  {"filename": "Northeast_Bank_fictitious_filled_loan_agreement.pdf", "chunks": 5},
  {"filename": "compliance_policy.md", "chunks": 17},
  {"filename": "filled_loan_agreement.pdf", "chunks": 10},
  {"filename": "glossary.md", "chunks": 29},
  {"filename": "product_faq.md", "chunks": 18},
  {"filename": "US_Employee_Referral_Policy.md", "chunks": 12}
]
```

**A:** I have access to 6 documents across PDF and Markdown formats, including two loan
agreements, a compliance policy, a product FAQ, a banking glossary, and an employee
referral policy.

---

### Q2 — Single-hop retrieval with citation

**Q:** *"What is the interest rate on the Northeast Bank loan agreement?"*

**Tools invoked:** `search_documents(query="interest rate APR", filename="Northeast_Bank_fictitious_filled_loan_agreement.pdf")`

**A:** Based on the Northeast Bank fictitious loan agreement, the Annual Percentage Rate
(APR) is **7.25%**. *(Source: Northeast_Bank_fictitious_filled_loan_agreement.pdf, p.1)*

---

### Q3 — Multi-document comparison (multi-hop)

**Q:** *"Compare the APR across both loan agreements and tell me which is cheaper."*

**Tools invoked:**
1. `extract_loan_terms(filename="Northeast_Bank_fictitious_filled_loan_agreement.pdf")` → `{"apr_pct": 7.25, "principal": 25000, "term_months": 60}`
2. `extract_loan_terms(filename="filled_loan_agreement.pdf")` → `{"apr_pct": 8.99, "principal": 18000, "term_months": 48}`

**A:** The Northeast Bank loan agreement carries an APR of **7.25%**, while the second loan
agreement has an APR of **8.99%**. The Northeast Bank loan is cheaper by **1.74 percentage
points**. Note that the lower APR combined with a longer term (60 vs. 48 months) means
total interest paid may differ — use the compute_loan_payment tool for a full cost comparison.

---

### Q4 — Tool chaining + deterministic math

**Q:** *"For the Northeast Bank loan, what would the monthly payment be if the principal were $50,000 instead?"*

**Tools invoked:**
1. `extract_loan_terms(filename="Northeast_Bank_fictitious_filled_loan_agreement.pdf")` → `{"apr_pct": 7.25, "term_months": 60}`
2. `compute_loan_payment(principal=50000, annual_rate_pct=7.25, term_months=60)`

**Tool output:**
```
Monthly payment: $994.94
  Principal: $50,000.00
  APR: 7.25%
  Term: 60 months
  Total paid: $59,696.40
  Total interest: $9,696.40
```

**A:** At a $50,000 principal with the same APR (7.25%) and term (60 months), the monthly
payment would be **$994.94**, with total interest of **$9,696.40** over the life of the loan.

---

### Q5 — Whole-document summarization

**Q:** *"Summarize the compliance policy in three bullets."*

**Tools invoked:** `summarize_document(filename="compliance_policy.md")`

**A:**
- **TILA/Regulation Z:** All loans require pre-consummation disclosure of APR, finance
  charge, amount financed, and total of payments; advertising specific rates must also
  prominently disclose the APR.
- **Fair lending (ECOA/Reg B):** Credit decisions must be free of discrimination on
  protected bases; adverse action notices with specific reasons are required within 30 days
  of a denial.
- **Subprime safeguards:** Borrowers below 620 FICO are subject to a 43% DTI cap,
  second-level approval, and prohibitions on balloon payments, negative amortization, and
  prepayment penalties.

---

### Q6 — Conversational memory (follow-up to Q5)

**Q (same thread):** *"Which of those points apply to subprime loans specifically?"*

**Tools invoked:** `search_documents(query="subprime loans requirements safeguards")`

**A:** From the compliance policy, the subprime-specific requirements are:
- DTI ratio must not exceed **43%** post-origination
- **Second-level manager review** is required for any underwriting exception
- Prohibited loan features: balloon payments, negative amortization, prepayment penalties,
  mandatory arbitration
- Quarterly portfolio review by the Credit Risk Committee
- Mandatory verbal disclosure of APR and total finance charge at application time
*(Source: compliance_policy.md > Subprime Lending Safeguards)*

*(This answer correctly references content from the prior turn's summary — demonstrating
LangGraph's MemorySaver thread-scoped conversational memory.)*

---

## Production trajectory

| POC component | Snowflake-native | AWS-native |
|---------------|-----------------|------------|
| PDF parsing | `AI_PARSE_DOCUMENT` (Cortex, handles OCR) | Amazon Textract |
| Chunking | `SPLIT_TEXT_RECURSIVE_CHARACTER` Cortex function | Lambda step in Step Functions |
| Embeddings | `AI_EMBED` with `snowflake-arctic-embed-m` | Bedrock `amazon.titan-embed-text-v2:0` |
| Vector store | `VECTOR(FLOAT, 768)` column + `VECTOR_COSINE_SIMILARITY`; or Cortex Search Service | OpenSearch Serverless; or Bedrock Knowledge Bases |
| Agent loop | Cortex Agents API; or LangGraph in Snowpark Container Services | Same LangGraph code with `ChatBedrockConverse`; or Bedrock AgentCore |
| Memory | Swap `MemorySaver` → `PostgresSaver` + Snowflake-fronted Postgres | `PostgresSaver` on RDS or DynamoDB |
| PII redaction | `AI_REDACT` Cortex function on chunks before embedding | Amazon Comprehend in ingest Lambda |
| Audit logging | `ACCOUNT_USAGE.QUERY_HISTORY` + custom event table | CloudTrail + CloudWatch |

The code migration to Bedrock is a **one-line swap**: replace `ChatAnthropic` with
`ChatBedrockConverse` in `src/agent/agent.py`. The graph, tools, memory, and checkpointer
are unchanged. AWS Bedrock AgentCore (launched 2025) supports LangGraph as a first-class
deployment target.

---

## What I'd add with more time

1. **PII redaction before embedding** — Cortex `AI_REDACT` or AWS Comprehend on every chunk; SSNs and account numbers should never become embeddings.
2. **Audit log every tool call** — wrap `@tool` functions or use a LangGraph callback to write `(session_id, tool_name, input_hash, output_hash, timestamp)` to an audit table. Required for SOX/OCC.
3. **Row-level access control** — filter ChromaDB queries by `branch_id` metadata so a loan officer in Branch A can't retrieve Branch B's loan chunks.
4. **Cross-encoder reranker** — `bge-reranker-base` or Cortex Search's built-in reranker on top of vector search; meaningful precision gain at low cost.
5. **Eval set** — 30–50 `(question, expected_filenames, expected_answer_substring)` tuples run on every change; LangSmith makes this turnkey for LangGraph apps.
6. **Streaming responses** — `agent.astream_events` piped to Streamlit's `st.write_stream` for perceived latency.
7. **Cost guardrails** — token spend per session logged, CloudWatch alarm on outliers. `max_iterations` cap already in place.
