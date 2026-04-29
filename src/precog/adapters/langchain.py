from __future__ import annotations

import json
from typing import Any, Mapping
from uuid import UUID

from ..runtime import PreCog


class PreCogLangChainCallbackHandler:
    """LangChain callback handler for observing tool results.

    LangChain callbacks cannot short-circuit a tool call, so this handler is for
    predictor training and memoization only. Use LangGraph's ToolNode wrapper
    when you need cache hits to skip execution.
    """

    def __init__(self, precog: PreCog) -> None:
        self.precog = precog
        self._runs: dict[str, tuple[str, dict[str, Any]]] = {}

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **_: Any,
    ) -> None:
        tool_name = str(serialized.get("name") or serialized.get("id") or "tool")
        args = _parse_tool_input(input_str)
        call_id = str(run_id)
        self._runs[call_id] = (tool_name, args)

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        **_: Any,
    ) -> None:
        call_id = str(run_id)
        stored = self._runs.pop(call_id, None)
        if stored is None:
            return
        tool_name, args = stored
        await self.precog.after_execute(tool_name, args, output, call_id=call_id)

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **_: Any,
    ) -> None:
        call_id = str(run_id)
        stored = self._runs.pop(call_id, None)
        if stored is None:
            return
        tool_name, args = stored
        await self.precog.after_execute(
            tool_name,
            args,
            repr(error),
            call_id=call_id,
            is_error=True,
        )


def _parse_tool_input(input_str: str) -> dict[str, Any]:
    try:
        parsed = json.loads(input_str)
    except json.JSONDecodeError:
        return {"input": input_str}
    if isinstance(parsed, Mapping):
        return dict(parsed)
    return {"input": parsed}
