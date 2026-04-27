"""LangGraph ReAct agent with MemorySaver-backed conversational memory.

Swap ChatAnthropic for ChatBedrockConverse (langchain-aws) to move to Bedrock
without changing any other code. The graph, tools, and checkpointer stay the same.
"""
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

from src.agent.tools import build_tools
from src.agent.prompts import SYSTEM_PROMPT
from src.config import settings
from src.store.vector_store import VectorStore


def get_llm():
    """Return a LangChain chat model based on LLM_PROVIDER setting."""
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.anthropic_api_key,
        )
    elif settings.llm_provider == "bedrock":
        from langchain_aws import ChatBedrockConverse
        return ChatBedrockConverse(
            model="anthropic.claude-sonnet-4-6",
            max_tokens=settings.llm_max_tokens,
        )
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")


def build_agent(store: VectorStore):
    """Construct the LangGraph ReAct agent.

    create_react_agent produces an explicit state-machine graph:
      agent node → (tool_calls?) → ToolNode → agent node → … → END
    The MemorySaver checkpointer persists state per thread_id so the same
    thread_id across turns gives multi-turn conversational memory.
    Swap MemorySaver for PostgresSaver / RedisSaver in production.
    """
    llm = get_llm()
    tools = build_tools(store)
    checkpointer = MemorySaver()

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    return agent


def chat_once(agent, user_message: str, thread_id: str) -> str:
    """Single conversational turn.

    Pass the same thread_id across calls and LangGraph's MemorySaver
    automatically restores prior conversation state for that thread.
    Different thread_ids are fully isolated — no cross-thread memory.
    """
    config = {
        "configurable": {"thread_id": thread_id},
        # recursion_limit = 2 × max_iterations because each tool round-trip
        # counts as two graph transitions (agent → tools → agent).
        "recursion_limit": settings.max_agent_iterations * 2,
    }
    result = agent.invoke(
        {"messages": [HumanMessage(content=user_message)]},
        config=config,
    )
    return result["messages"][-1].content


def stream_events(agent, user_message: str, thread_id: str):
    """Stream agent execution events for the UI tool-trace panel.

    Yields dicts from agent.stream(..., stream_mode="updates").
    Each dict is keyed by node name ("agent" or "tools") with the node's output.
    """
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": settings.max_agent_iterations * 2,
    }
    yield from agent.stream(
        {"messages": [HumanMessage(content=user_message)]},
        config=config,
        stream_mode="updates",
    )


def export_graph_diagram(agent, output_path: str = "docs/agent_graph.png") -> None:
    """Render the agent's state-machine graph as a PNG via Mermaid.

    Requires the mermaid-py package or access to the Mermaid CLI.
    Drop the output image into the README as visual proof of agentic structure.
    """
    png_bytes = agent.get_graph().draw_mermaid_png()
    with open(output_path, "wb") as f:
        f.write(png_bytes)
