from __future__ import annotations

"""Skeleton for wiring PreCog into Anthropic/Claude Messages tool use."""

import asyncio
import json

from precog import IdempotencyClass, PreCog, ToolRegistry
from precog.adapters.anthropic import execute_message_tool_calls


registry = ToolRegistry()


@registry.register(idempotency_class=IdempotencyClass.NETWORK_READ)
def get_order_status(order_id: str) -> dict[str, str]:
    """Fetch an order's current status."""

    return {"order_id": order_id, "status": "in_transit"}


async def main() -> None:
    precog = PreCog(**registry.precog_kwargs())

    # This mirrors an Anthropic Messages response containing a tool_use block.
    message = {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_order_status",
                "input": {"order_id": "ord_123"},
            }
        ],
        "stop_reason": "tool_use",
    }

    tool_results = await execute_message_tool_calls(
        precog,
        message,
        registry.execute,
    )

    print("tools:")
    print(json.dumps(registry.anthropic_tools(), indent=2))
    print("next user content:")
    print(json.dumps(tool_results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
