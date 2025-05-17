"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import json
from langchain_core.messages import AIMessage

# from openai import OpenAI
from openai import OpenAI

embeddings = OpenAIEmbeddings()
client = OpenAI()

vector_store = client.vector_stores.create(
    name="knowledge_base"
)

# Clear the vector store
file_list = client.files.list()
for file in file_list:
    client.files.delete(
        file_id=file.id,
    )

try:
    with open("data/deep_research_blog.pdf", "rb") as file_content:
        result = client.files.create(
            file=file_content,
            purpose="assistants"
        )
        client.vector_stores.files.create(
            vector_store_id=vector_store.id,
            file_id=result.id,
        )
    
except OSError as e:
    print("Error uploading file:", e)

class Configuration(TypedDict):
    """Configurable parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    my_configurable_param: str


@dataclass
class State:
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    user_input: str

@dataclass
class StateOutput:
    """Output state for the agent.

    Defines the structure of outgoing data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    response: str
    annotations: dict[str, Any]

llm = ChatOpenAI(model="gpt-4o-mini")

openai_vector_store_ids = [
    vector_store.id,
]

tool = {
    "type": "file_search",
    "vector_store_ids": openai_vector_store_ids,
}
llm_with_tools = llm.bind_tools([tool])

async def call_model(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Process input and returns output.

    Can use runtime configuration to alter behavior.
    """
    configuration = config["configurable"]

    response = await llm_with_tools.ainvoke(state.user_input)
    
    return {
        "response": response.text(),
        "annotations": response.content[0]["annotations"][0],
    }


# Define the graph
graph = (
    StateGraph(input=State, config_schema=Configuration, output=StateOutput)
    .add_node(call_model)
    .add_edge(START, "call_model")
    .add_edge("call_model", END)
    .compile(name="New Graph")
)
