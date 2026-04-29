from __future__ import annotations

import asyncio
import json

from precog import IdempotencyClass, PreCog, ToolRegistry


registry = ToolRegistry()


@registry.register(idempotency_class=IdempotencyClass.NETWORK_READ, timeout_seconds=2)
def search(q: str) -> dict[str, str]:
    return {"query": q, "result": f"cached result for {q}"}


async def main() -> None:
    precog = PreCog(**registry.precog_kwargs())
    args = {"q": "agent latency"}

    await precog.observe_model_event(
        {"type": "tool_call_start", "callId": "call-1", "toolName": "search"}
    )
    await precog.observe_model_event(
        {
            "type": "tool_call_args_delta",
            "callId": "call-1",
            "delta": json.dumps(args),
        }
    )
    await precog.observe_model_event({"type": "tool_call_end", "callId": "call-1"})

    result = await precog.execute("search", args, registry.execute, call_id="call-1")
    await precog.close()

    print(json.dumps(result, indent=2))
    print(json.dumps(precog.snapshot(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
