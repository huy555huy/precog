from __future__ import annotations

"""Skeleton for wiring PreCog into an OpenAI Responses tool loop.

This file intentionally avoids making a network request. Drop in an OpenAI
client and pass `registry.openai_tools()` to `client.responses.create(...)`.
"""

import asyncio
import json

from precog import IdempotencyClass, PreCog, ToolRegistry
from precog.adapters.openai import execute_response_tool_calls


registry = ToolRegistry()


@registry.register(idempotency_class=IdempotencyClass.NETWORK_READ)
def get_order_status(order_id: str) -> dict[str, str]:
    """Fetch an order's current status."""

    return {"order_id": order_id, "status": "in_transit"}


async def main() -> None:
    precog = PreCog(**registry.precog_kwargs())

    # This mirrors a Responses API object containing a function_call item.
    response = {
        "output": [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "get_order_status",
                "arguments": json.dumps({"order_id": "ord_123"}),
            }
        ]
    }

    tool_outputs = await execute_response_tool_calls(
        precog,
        response,
        registry.execute,
    )

    print("tools:")
    print(json.dumps(registry.openai_tools(), indent=2))
    print("next input items:")
    print(json.dumps(tool_outputs, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
