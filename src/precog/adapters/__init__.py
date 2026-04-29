"""Optional integrations for popular agent runtimes."""

from .anthropic import (
    AnthropicMessagesAdapter,
    AnthropicMessagesClient,
    AnthropicStreamAccumulator,
    AnthropicToolUse,
    execute_message_tool_calls,
    extract_tool_uses,
    tool_result_block,
)
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
    "AnthropicMessagesAdapter",
    "AnthropicMessagesClient",
    "AnthropicStreamAccumulator",
    "AnthropicToolUse",
    "OpenAIResponsesAdapter",
    "PreCogLangChainCallbackHandler",
    "events_from_openai_response_event",
    "execute_message_tool_calls",
    "execute_response_tool_calls",
    "extract_function_calls",
    "extract_tool_uses",
    "function_call_output",
    "make_langgraph_tool_wrappers",
    "tool_result_block",
]
