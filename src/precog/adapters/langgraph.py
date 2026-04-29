from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from ..runtime import PreCog


def make_langgraph_tool_wrappers(precog: PreCog) -> dict[str, Any]:
    """Return ToolNode wrapper kwargs for LangGraph.

    Usage:
        ToolNode(tools, **make_langgraph_tool_wrappers(precog))
    """

    def wrap_tool_call(request: Any, execute: Callable[[Any], Any]) -> Any:
        tool_name, args, call_id = _request_parts(request)
        decision = _run_blocking(precog.before_execute(tool_name, args, call_id=call_id))
        if decision.type == "provide_result":
            return decision.result
        try:
            result = execute(request)
        except Exception:
            _run_blocking(
                precog.after_execute(
                    tool_name,
                    args,
                    None,
                    call_id=call_id,
                    is_error=True,
                )
            )
            raise
        _run_blocking(precog.after_execute(tool_name, args, result, call_id=call_id))
        return result

    async def awrap_tool_call(
        request: Any,
        execute: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        tool_name, args, call_id = _request_parts(request)
        decision = await precog.before_execute(tool_name, args, call_id=call_id)
        if decision.type == "provide_result":
            return decision.result
        try:
            result = await execute(request)
        except Exception:
            await precog.after_execute(
                tool_name,
                args,
                None,
                call_id=call_id,
                is_error=True,
            )
            raise
        await precog.after_execute(tool_name, args, result, call_id=call_id)
        return result

    return {"wrap_tool_call": wrap_tool_call, "awrap_tool_call": awrap_tool_call}


def _request_parts(request: Any) -> tuple[str, dict[str, Any], str]:
    tool_call = getattr(request, "tool_call", request)
    name = tool_call["name"]
    args = dict(tool_call.get("args", {}))
    call_id = str(tool_call.get("id") or tool_call.get("call_id") or name)
    return str(name), args, call_id


def _run_blocking(awaitable: Awaitable[Any]) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    raise RuntimeError(
        "sync LangGraph wrapper cannot run inside an active event loop; "
        "use awrap_tool_call instead"
    )
