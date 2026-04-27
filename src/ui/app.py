"""Streamlit chat UI for the Northeast Bank Agentic Document Q&A system.

Run with:
    streamlit run src/ui/app.py

Features:
- Upload PDF/MD files from the browser → triggers ingestion pipeline
- Chat interface backed by LangGraph ReAct agent with MemorySaver memory
- Sidebar showing all ingested documents and chunk counts
- Expandable tool-call trace under each assistant message
- "Show agent graph" button renders the LangGraph state-machine diagram
"""
import sys
from pathlib import Path

# Ensure the project root is on sys.path regardless of how streamlit is invoked.
# __file__ = src/ui/app.py → .parent.parent.parent = project root
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import json
import uuid
import tempfile

import streamlit as st

from src.store.vector_store import VectorStore
from src.agent.agent import build_agent, stream_events
from src.ingestion.pipeline import ingest_file


# ── Streamlit page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="NEB Document Q&A",
    page_icon="🏦",
    layout="wide",
)


# ── Session-state helpers ────────────────────────────────────────────────────
def init_session() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = uuid.uuid4().hex
    if "messages" not in st.session_state:
        # Each entry: {"role": "user"|"assistant", "content": str, "tool_events": list}
        st.session_state.messages = []
    if "store" not in st.session_state:
        st.session_state.store = VectorStore()
    if "agent" not in st.session_state:
        st.session_state.agent = build_agent(st.session_state.store)


def reset_conversation() -> None:
    st.session_state.thread_id = uuid.uuid4().hex
    st.session_state.messages = []


# ── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar(store: VectorStore) -> None:
    with st.sidebar:
        st.title("🏦 NEB Doc Q&A")
        st.markdown("---")

        # File uploader
        st.subheader("Upload documents")
        uploaded = st.file_uploader(
            "Drag PDF or Markdown files here",
            type=["pdf", "md"],
            accept_multiple_files=True,
            key="file_uploader",
        )
        if uploaded:
            for uf in uploaded:
                suffix = Path(uf.name).suffix
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix, prefix=Path(uf.name).stem + "_"
                ) as tmp:
                    tmp.write(uf.getbuffer())
                    tmp_path = Path(tmp.name)

                with st.spinner(f"Ingesting {uf.name}…"):
                    result = ingest_file(tmp_path, store)
                    result["file"] = uf.name  # show original name

                if result["status"] == "ingested":
                    st.success(f"{uf.name}: {result['chunks']} chunks ingested")
                elif result["status"] == "skipped_duplicate":
                    st.info(f"{uf.name}: already in knowledge base")
                else:
                    st.warning(f"{uf.name}: {result['status']}")

        st.markdown("---")

        # Ingested documents list
        st.subheader("Knowledge base")
        files = store.list_files()
        if files:
            for f in files:
                st.markdown(f"**{f['filename']}** — {f['chunks']} chunks")
        else:
            st.caption("No documents ingested yet. Upload files above or run `python ingest.py docs/`.")

        st.markdown("---")

        # Agent graph diagram
        if st.button("Show agent graph"):
            agent = st.session_state.agent
            try:
                png = agent.get_graph().draw_mermaid_png()
                st.image(png, caption="LangGraph state machine")
            except Exception as e:
                st.error(f"Could not render graph: {e}")

        st.markdown("---")

        # Conversation controls
        if st.button("New conversation"):
            reset_conversation()
            st.rerun()

        st.caption(f"Thread: `{st.session_state.thread_id[:8]}…`")


# ── Chat rendering ────────────────────────────────────────────────────────────
def render_tool_events(events: list[dict]) -> None:
    """Render a collapsible tool-trace panel from stream_mode='updates' events."""
    tool_calls = []
    for event in events:
        # "tools" node returns ToolMessage objects
        tool_node_output = event.get("tools", {})
        messages = tool_node_output.get("messages", [])
        for msg in messages:
            tool_calls.append({
                "tool": getattr(msg, "name", "?"),
                "content": msg.content if hasattr(msg, "content") else str(msg),
            })
        # "agent" node may contain AIMessage with tool_calls
        agent_output = event.get("agent", {})
        for msg in agent_output.get("messages", []):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    # store call input alongside the result recorded above
                    if tool_calls and tool_calls[-1]["tool"] == tc.get("name"):
                        tool_calls[-1]["input"] = tc.get("args", {})

    if tool_calls:
        with st.expander(f"🔧 Tool calls ({len(tool_calls)})", expanded=False):
            for tc in tool_calls:
                st.markdown(f"**`{tc['tool']}`**")
                if "input" in tc:
                    st.json(tc["input"])
                output = tc.get("content", "")
                if len(output) > 600:
                    output = output[:600] + "\n…[truncated]"
                st.code(output, language="text")


def render_chat_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("tool_events"):
                render_tool_events(msg["tool_events"])


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    init_session()
    store: VectorStore = st.session_state.store
    agent = st.session_state.agent

    render_sidebar(store)

    st.header("Document Q&A")
    st.caption(
        "Ask questions about the ingested documents. "
        "The agent will search, extract, and calculate as needed."
    )

    render_chat_history()

    user_input = st.chat_input("Ask a question about your documents…")
    if not user_input:
        return

    # Display user message immediately
    st.session_state.messages.append(
        {"role": "user", "content": user_input, "tool_events": []}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream agent response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        tool_events: list[dict] = []
        final_content = ""

        with st.spinner("Thinking…"):
            for event in stream_events(agent, user_input, st.session_state.thread_id):
                tool_events.append(event)
                # Extract final answer text from agent node messages
                agent_msgs = event.get("agent", {}).get("messages", [])
                for msg in agent_msgs:
                    content = getattr(msg, "content", "")
                    if content and not getattr(msg, "tool_calls", None):
                        final_content = content

        response_placeholder.markdown(final_content or "_No response generated._")
        if tool_events:
            render_tool_events(tool_events)

    st.session_state.messages.append({
        "role": "assistant",
        "content": final_content,
        "tool_events": tool_events,
    })


if __name__ == "__main__":
    main()
