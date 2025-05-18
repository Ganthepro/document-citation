from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from agent.state import InputState, OutputState, ConversationState
from agent.utils import vector_store_setup
from langgraph.graph.message import RemoveMessage, add_messages
from agent.configuration import Configuration
from typing import Literal

vector_store = vector_store_setup()


async def summarize_conversation(
    state: ConversationState, config: RunnableConfig
) -> dict:
    """Summarize the conversation and return the summary."""

    summary = state.get("summary", "")
    # Create our summarization prompt
    if summary:
        # A summary already exists
        summary_message = (
            f"This is a summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )

    else:
        summary_message = "Create a summary of the conversation above:"

    configuration = config["configurable"]
    llm = ChatOpenAI(model=configuration.get("model_name", "gpt-4o-mini"))

    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = await llm.ainvoke(messages)

    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}


async def call_model_with_memory(
    conversation_state: ConversationState,
    config: RunnableConfig,
) -> ConversationState:
    """Process input with memory and returns output."""
    configuration = config["configurable"]
    llm = ChatOpenAI(model=configuration.get("model_name", "gpt-4o-mini"))

    openai_vector_store_ids = [
        vector_store.id,
    ]

    tool = {
        "type": "file_search",
        "vector_store_ids": openai_vector_store_ids,
    }
    llm_with_tools = llm.bind_tools([tool])

    # Add the current input to the chat history
    current_message = HumanMessage(content=conversation_state["user_input"])
    summary_message = AIMessage(
        content=conversation_state.get("summary", "")
    )

    # Create the message list with history and current message
    print("Conversation state messages:", conversation_state["messages"])
    messages = conversation_state["messages"] + [current_message] + [summary_message]

    # Get the response from the model with tools and history
    response = await llm_with_tools.ainvoke(messages)

    # Extract the response text and annotations
    response_text = response.text()
    annotations = (
        response.content[0].get("annotations", None) if response.content else None
    )

    # Create an AI message from the response
    ai_message = AIMessage(content=response_text)

    return ConversationState(
        messages=add_messages(
            conversation_state["messages"],
            [current_message, ai_message],
        ),
        user_input=conversation_state["user_input"],
        response=response_text,
        annotations=annotations,
    )


def is_summary_needed(
    conversation_state: ConversationState,
    config: RunnableConfig,
) -> Literal["summary", "__end__"]:
    """Determine if a summary is needed based on the conversation state."""

    # Check if the summary is empty or if the conversation has reached a certain length
    if len(conversation_state["messages"]) > 5:
        return "summary"
    return "__end__"


def create_graph():
    """Create the graph with memory capability."""
    graph = (
        StateGraph(
            ConversationState,
            input=InputState,
            output=OutputState,
            config_schema=Configuration,
        )
        .add_node("call_model", call_model_with_memory)
        .add_node("summarize_conversation", summarize_conversation)
        .add_edge(START, "call_model")
        .add_conditional_edges(
            "call_model",
            is_summary_needed,
            {
                "summary": "summarize_conversation",
                "__end__": END,
            },
        )
        .add_edge("summarize_conversation", END)
        # .add_edge("summarize_conversation", "call_model")
        
    )

    return graph.compile(name="Document Citation Agent With Memory")


# Create the graph
graph = create_graph()
