"""Optional integrations for popular agent runtimes."""

from .langchain import PreCogLangChainCallbackHandler
from .langgraph import make_langgraph_tool_wrappers
from .openai import (
    OpenAIFunctionCall,
    OpenAIResponsesAdapter,
    events_from_openai_response_event,
    execute_response_tool_calls,
    extract_function_calls,
    function_call_output,
)

__all__ = [
    "OpenAIFunctionCall",
    "OpenAIResponsesAdapter",
    "PreCogLangChainCallbackHandler",
    "events_from_openai_response_event",
    "execute_response_tool_calls",
    "extract_function_calls",
    "function_call_output",
    "make_langgraph_tool_wrappers",
]
