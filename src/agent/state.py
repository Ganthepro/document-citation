from typing import Any, Optional, TypedDict
from langgraph.graph import MessagesState

class InputState(TypedDict):
    """State that contains the user input."""
    
    # The current input from the user
    user_input: str


class OutputState(TypedDict):
    """State that contains the output from the model."""
    
    # The response to return to the user
    response: Optional[str] = None
    
    # Any annotations from the model
    annotations: Optional[dict[str, Any]] = None


class ConversationState(MessagesState):
    """State that contains the summary of the conversation."""
    
    # The summary of the conversation
    summary: str = ""

    user_input: str

    response: Optional[str] = None

    annotations: Optional[dict[str, Any]] = None
