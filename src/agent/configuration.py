from dataclasses import dataclass, field


@dataclass
class Configuration:
    """Configurable parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    # The model to use for the agent.
    model_name: str = field(default="gpt-4o-mini")