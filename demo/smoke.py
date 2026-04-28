from __future__ import annotations

import asyncio
import json
from typing import Any, Mapping

from precog import PreCog


async def executor(tool_name: str, args: Mapping[str, Any]) -> dict[str, Any]:
    await asyncio.sleep(0.02)
    return {"tool": tool_name, "args": dict(args), "source": "speculative"}


async def main() -> None:
    precog = PreCog(read_only_tools={"search"}, executor=executor, verbose=True)
    call_id = "call-1"
    args = {"q": "agent runtime architecture"}

    await precog.observe_model_event(
        {"type": "tool_call_start", "callId": call_id, "toolName": "search"}
    )
    await precog.observe_model_event(
        {
            "type": "tool_call_args_delta",
            "callId": call_id,
            "delta": json.dumps(args),
        }
    )
    await precog.observe_model_event({"type": "tool_call_end", "callId": call_id})

    decision = await precog.before_execute("search", args, call_id=call_id)
    print("decision:", decision.type)
    print("result:", decision.result)
    print(json.dumps(precog.snapshot(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())

