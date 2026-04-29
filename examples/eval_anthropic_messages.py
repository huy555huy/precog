from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping

from precog import IdempotencyClass, PreCog, ToolRegistry
from precog.adapters.anthropic import AnthropicMessagesClient, execute_message_tool_calls


TASKS = [
    "Check order ord_100, then check it again and summarize whether the two results match.",
    "Find docs for refund policy twice, then explain the answer with citations from the tool outputs.",
    "Look up account acct_42, then check order ord_100 again and summarize both.",
]


@dataclass
class EvalResult:
    mode: str
    task_count: int
    wall_time_ms: float
    tool_executions: int
    stats: dict[str, Any]


def build_registry(latency_ms: int) -> tuple[ToolRegistry, dict[str, int]]:
    registry = ToolRegistry()
    counters = {"executions": 0}

    def sleep() -> None:
        counters["executions"] += 1
        time.sleep(latency_ms / 1000)

    @registry.register(idempotency_class=IdempotencyClass.NETWORK_READ)
    def get_order_status(order_id: str) -> dict[str, str]:
        """Get the current status for an order id."""

        sleep()
        return {"order_id": order_id, "status": "in_transit", "eta": "tomorrow"}

    @registry.register(idempotency_class=IdempotencyClass.NETWORK_READ)
    def search_docs(query: str) -> dict[str, Any]:
        """Search support documentation for a user-facing policy answer."""

        sleep()
        return {
            "query": query,
            "results": [
                {
                    "doc_id": "refund_policy",
                    "snippet": "Refunds are available within 30 days for unopened items.",
                }
            ],
        }

    @registry.register(idempotency_class=IdempotencyClass.NETWORK_READ)
    def get_account(account_id: str) -> dict[str, str]:
        """Fetch a customer account profile."""

        sleep()
        return {"account_id": account_id, "tier": "pro", "region": "us"}

    return registry, counters


async def run_mode(args: argparse.Namespace, mode: str) -> EvalResult:
    registry, counters = build_registry(args.tool_latency_ms)
    precog = PreCog(**registry.precog_kwargs(), rollout_mode=_rollout_mode(mode))
    client = AnthropicMessagesClient(model=args.model) if not args.dry_run else None

    started = time.monotonic()
    for task in TASKS[: args.task_limit]:
        if args.dry_run:
            continue
        await run_task(client, precog, registry, task, args.max_rounds, args.max_tokens)
    wall_time_ms = (time.monotonic() - started) * 1000

    return EvalResult(
        mode=mode,
        task_count=args.task_limit,
        wall_time_ms=wall_time_ms,
        tool_executions=counters["executions"],
        stats=precog.snapshot(),
    )


async def run_task(
    client: AnthropicMessagesClient,
    precog: PreCog,
    registry: ToolRegistry,
    task: str,
    max_rounds: int,
    max_tokens: int,
) -> None:
    messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
    system = (
        "Use the provided tools for every factual lookup, even if you think you "
        "know the answer. Reuse tools when the task asks you to check twice."
    )

    for _ in range(max_rounds):
        response = client.messages_create(
            max_tokens=max_tokens,
            system=system,
            tools=registry.anthropic_tools(),
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.get("content", [])})
        if response.get("stop_reason") != "tool_use":
            return
        tool_results = await execute_message_tool_calls(precog, response, registry.execute)
        messages.append({"role": "user", "content": tool_results})


def _rollout_mode(mode: str) -> str:
    if mode == "baseline":
        return "off"
    if mode == "observe":
        return "observe"
    if mode == "memoize":
        return "memoize"
    if mode == "speculate":
        return "speculate"
    raise ValueError(f"unknown mode: {mode}")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["baseline", "observe", "memoize", "speculate", "all"],
        default="all",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--task-limit", type=int, default=len(TASKS))
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--tool-latency-ms", type=int, default=250)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    modes = ["baseline", "observe", "memoize", "speculate"] if args.mode == "all" else [args.mode]
    results = [await run_mode(args, mode) for mode in modes]
    payload = {"results": [asdict(result) for result in results]}

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"anthropic_eval_{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path = out_dir / f"anthropic_eval_{stamp}.md"
    md_path.write_text(_markdown_report(results), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


def _markdown_report(results: list[EvalResult]) -> str:
    lines = [
        "# Anthropic Messages Eval",
        "",
        "| mode | tasks | wall ms | tool exec | hit rate | cache hits | shadow hits | wasted |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        stats = result.stats["stats"]
        lines.append(
            "| "
            f"{result.mode} | {result.task_count} | {result.wall_time_ms:.0f} | "
            f"{result.tool_executions} | {result.stats['hit_rate']:.2f} | "
            f"{stats['cache_hits']} | {stats['shadow_hits']} | {stats['wasted_speculations']} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
