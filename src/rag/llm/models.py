from langchain_openai.chat_models import ChatOpenAI
from langchain.chat_models import BaseChatModel

from enum import Enum
from typing import Optional,Annotated, TypedDict, Union, Callable

from dotenv import load_dotenv
from os import getenv

load_dotenv()

class LLMProvider :
    class Cloud(Enum) :
        GOOGLE_CHAT = "google_chat"
        OPEN_AI = "openai"
        DEEPSEEK = "deepseek"
    class Local(Enum):
        OLLAMA = "ollama"
        LM_STUDIO = "lm_studio"

def get_llm_api_key(llm: Union[LLMProvider.Cloud, LLMProvider.Local]) -> Callable[[], str]:
    if llm == LLMProvider.Cloud.GOOGLE_CHAT:
        return lambda: getenv("GOOGLE_API_KEY", "")
    elif llm == LLMProvider.Cloud.OPEN_AI:
        return lambda: getenv("OPENAI_API_KEY", "")
    elif llm == LLMProvider.Cloud.DEEPSEEK:
        return lambda: getenv("DEEPSEEK_API_KEY", "")
    elif llm == LLMProvider.Local.OLLAMA:
        return lambda: ""
    elif llm == LLMProvider.Local.LM_STUDIO:
        return lambda: ""
    else:
        raise ValueError(f"Unsupported LLM provider: {llm}")
    
def get_llm_model(llm_provider: Union[LLMProvider.Cloud, LLMProvider.Local], model_name: str) -> BaseChatModel:
    if llm_provider == LLMProvider.Cloud.GOOGLE_CHAT:
        llm = ChatOpenAI(
            model = model_name,
            base_url="https://generativelanguage.googleapis.com/v1beta3/models/",
            api_key=get_llm_api_key(llm_provider),
            temperature=0.7,
        )
        return llm
    elif llm_provider == LLMProvider.Cloud.OPEN_AI:
        llm = ChatOpenAI(
            model = model_name,
            base_url="https://api.openai.com/v1/",
            api_key=get_llm_api_key(llm_provider),
            temperature=0.7,
        )
        return llm
    elif llm_provider == LLMProvider.Cloud.DEEPSEEK:
        llm = ChatOpenAI(
            model = model_name,
            base_url="https://api.deepseek.com/v1/",
            api_key=get_llm_api_key(llm_provider),
            temperature=0.7,
        )
        return llm
    elif llm_provider == LLMProvider.Local.OLLAMA:
        llm = ChatOpenAI(
            model = model_name,
            base_url = "http://localhost:11434/api/v1/",
            api_key=get_llm_api_key(llm_provider),
            temperature=0.7,
        )
        return llm
    elif llm_provider == LLMProvider.Local.LM_STUDIO:
        llm = ChatOpenAI(
            model = model_name,
            base_url = "http://localhost:8080/api/v1/",
            api_key=get_llm_api_key(llm_provider),
            temperature=0.7,
        )
        return llm
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")

