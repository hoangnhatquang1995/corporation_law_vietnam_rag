from settings.types import StateNode
from rag.llm.models import get_llm_model, LLMProvider
from rag.llm import llm
from .system_prompts import llm_system_prompt

def llm_node(state: StateNode) -> StateNode:
    messages = [
        llm_system_prompt,
        *state["messages"]
    ]
    response = llm.invoke(messages)
    return {
        "messages": [response]
    }




