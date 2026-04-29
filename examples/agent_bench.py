from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping

from precog import IdempotencyClass, PreCog, ToolRegistry
from precog.adapters.anthropic import (
    AnthropicMessagesAdapter,
    AnthropicMessagesClient,
    execute_message_tool_calls,
    extract_tool_uses,
)


SYSTEM_PROMPT = """
You are a deterministic support operations agent.
Use the provided tools for every factual lookup.
If a task says to check or search something twice, make two separate tool calls
with identical arguments before you answer. Do not invent facts from memory.
When done, answer in compact JSON with an "answer" string and an "evidence"
array of facts copied from tool outputs.
""".strip()


@dataclass(frozen=True)
class AgentTask:
    name: str
    prompt: str
    must_include: tuple[str, ...]


@dataclass
class AgentBenchResult:
    mode: str
    task_count: int
    passed: int
    wall_time_ms: float
    tool_calls_requested: int
    tool_executions: int
    stats: dict[str, Any]


TASKS = [
    AgentTask(
        name="repeat_order_lookup",
        prompt=(
            "Check order ord_100 twice with identical tool arguments. Then say "
            "whether the two checks agree, including the order id, status, and ETA."
        ),
        must_include=("ord_100", "in_transit", "tomorrow"),
    ),
    AgentTask(
        name="repeat_policy_search",
        prompt=(
            "Search the support docs for refund policy twice with identical tool "
            "arguments. Then cite the policy id and the refund window."
        ),
        must_include=("refund_policy", "30 days", "unopened"),
    ),
    AgentTask(
        name="mixed_account_order",
        prompt=(
            "Look up account acct_42 twice with identical tool arguments, then "
            "check order ord_100 once. Summarize the account tier, region, and "
            "order status."
        ),
        must_include=("acct_42", "pro", "us", "in_transit"),
    ),
]


def build_registry(latency_ms: int) -> tuple[ToolRegistry, dict[str, Any]]:
    registry = ToolRegistry()
    counters: dict[str, Any] = {"executions": 0, "by_tool": {}}

    def sleep(tool_name: str) -> None:
        counters["executions"] += 1
        by_tool = counters["by_tool"]
        by_tool[tool_name] = by_tool.get(tool_name, 0) + 1
        time.sleep(latency_ms / 1000)

    @registry.register(idempotency_class=IdempotencyClass.NETWORK_READ)
    def get_order_status(order_id: str) -> dict[str, str]:
        """Fetch the current status for a customer order id."""

        sleep("get_order_status")
        orders = {
            "ord_100": {
                "order_id": "ord_100",
                "status": "in_transit",
                "eta": "tomorrow",
            }
        }
        return orders.get(order_id, {"order_id": order_id, "status": "unknown"})

    @registry.register(idempotency_class=IdempotencyClass.NETWORK_READ)
    def search_policy(query: str) -> dict[str, Any]:
        """Search internal support policy documents by query string."""

        sleep("search_policy")
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
        """Fetch a customer account profile by account id."""

        sleep("get_account")
        accounts = {
            "acct_42": {"account_id": "acct_42", "tier": "pro", "region": "us"}
        }
        return accounts.get(account_id, {"account_id": account_id, "tier": "unknown"})

    return registry, counters


async def run_mode(args: argparse.Namespace, mode: str) -> AgentBenchResult:
    registry, counters = build_registry(args.tool_latency_ms)
    rollout_mode = _rollout_mode(mode)
    precog_kwargs = registry.precog_kwargs()
    precog_kwargs["executor"] = registry.execute if rollout_mode == "speculate" else None
    precog = PreCog(
        **precog_kwargs,
        rollout_mode=rollout_mode,
    )
    client = AnthropicMessagesClient(model=args.model) if not args.dry_run else None

    passed = 0
    tool_calls_requested = 0
    started = time.monotonic()

    for task in TASKS[: args.task_limit]:
        if args.dry_run:
            continue
        outcome = await run_agent_task(
            client=client,
            precog=precog,
            registry=registry,
            task=task,
            max_rounds=args.max_rounds,
            max_tokens=args.max_tokens,
            stream=not args.no_stream,
        )
        passed += int(outcome["passed"])
        tool_calls_requested += int(outcome["tool_calls_requested"])

    wall_time_ms = (time.monotonic() - started) * 1000
    return AgentBenchResult(
        mode=mode,
        task_count=args.task_limit,
        passed=passed,
        wall_time_ms=wall_time_ms,
        tool_calls_requested=tool_calls_requested,
        tool_executions=int(counters["executions"]),
        stats=precog.snapshot(),
    )


async def run_agent_task(
    *,
    client: AnthropicMessagesClient,
    precog: PreCog,
    registry: ToolRegistry,
    task: AgentTask,
    max_rounds: int,
    max_tokens: int,
    stream: bool,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [{"role": "user", "content": task.prompt}]
    tool_calls_requested = 0
    final_text = ""

    for _ in range(max_rounds):
        if stream:
            adapter = AnthropicMessagesAdapter()

            async def on_event(event: dict[str, Any]) -> None:
                for model_event in adapter.events_from(event):
                    await precog.observe_model_event(model_event)

            response = await client.messages_create_streaming(
                on_event=on_event,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                tools=registry.anthropic_tools(),
                messages=messages,
            )
        else:
            response = client.messages_create(
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                tools=registry.anthropic_tools(),
                messages=messages,
            )

        messages.append({"role": "assistant", "content": response.get("content", [])})
        final_text = _message_text(response)
        tool_uses = extract_tool_uses(response)

        if response.get("stop_reason") != "tool_use" and not tool_uses:
            return {
                "passed": _passes(final_text, task.must_include),
                "tool_calls_requested": tool_calls_requested,
                "final_text": final_text,
            }

        tool_calls_requested += len(tool_uses)
        tool_results = await execute_message_tool_calls(
            precog,
            response,
            registry.execute,
        )
        messages.append({"role": "user", "content": tool_results})

    return {
        "passed": False,
        "tool_calls_requested": tool_calls_requested,
        "final_text": final_text,
    }


def _message_text(message: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for block in message.get("content", []) or []:
        if isinstance(block, Mapping) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(parts)


def _passes(text: str, required: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return all(token.lower() in lowered for token in required)


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


def _selected_modes(raw: str) -> list[str]:
    if raw == "all":
        return ["baseline", "observe", "memoize", "speculate"]
    modes = [mode.strip() for mode in raw.split(",") if mode.strip()]
    allowed = {"baseline", "observe", "memoize", "speculate"}
    invalid = [mode for mode in modes if mode not in allowed]
    if invalid:
        raise ValueError(f"unknown mode(s): {', '.join(invalid)}")
    return modes


def markdown_report(results: list[AgentBenchResult], args: argparse.Namespace) -> str:
    lines = [
        "# PreCog Agent Bench",
        "",
        f"- model: `{args.model or 'env/default'}`",
        f"- tasks: `{args.task_limit}`",
        f"- tool latency: `{args.tool_latency_ms}ms`",
        f"- streaming: `{not args.no_stream}`",
        "",
        "| mode | pass | wall ms | tool calls | tool exec | cache hits | "
        "shadow hits | spec resolved | saved ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        stats = result.stats["stats"]
        lines.append(
            "| "
            f"{result.mode} | {result.passed}/{result.task_count} | "
            f"{result.wall_time_ms:.0f} | {result.tool_calls_requested} | "
            f"{result.tool_executions} | {stats['cache_hits']} | "
            f"{stats['shadow_hits']} | {stats['speculations_resolved']} | "
            f"{stats['latency_saved_ms']:.0f} |"
        )
    lines.append("")
    return "\n".join(lines)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default="all")
    parser.add_argument("--model", default=None)
    parser.add_argument("--task-limit", type=int, default=len(TASKS))
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--tool-latency-ms", type=int, default=600)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()
    args.task_limit = max(0, min(args.task_limit, len(TASKS)))

    modes = _selected_modes(args.modes)
    results = [await run_mode(args, mode) for mode in modes]
    payload = {"results": [asdict(result) for result in results]}

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"agent_bench_{stamp}.json"
    md_path = out_dir / f"agent_bench_{stamp}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(markdown_report(results, args), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
