from typing import Annotated,TypedDict, List, Optional, Union
from langgraph.graph.message import add_messages
from langchain.messages import AnyMessage

class StateNode(TypedDict):
    messages : Annotated[List[AnyMessage], add_messages]
