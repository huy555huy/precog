from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import time
from typing import Any, Mapping

from precog import PreCog


READ_ONLY_TOOLS = {"search", "fetch_url", "read_file"}


@dataclass(frozen=True)
class TrajectoryStep:
    call_id: str
    tool_name: str
    args: Mapping[str, Any]
    exec_ms: int
    stream_ms: int
    think_ms: int


def build_trajectory() -> list[TrajectoryStep]:
    raw = [
        ("search", {"q": "agent runtime architecture"}, 320, 220, 600),
        ("fetch_url", {"url": "https://a.example"}, 800, 200, 400),
        ("search", {"q": "agent runtime architecture"}, 320, 220, 600),
        ("fetch_url", {"url": "https://a.example"}, 800, 200, 400),
        ("write_file", {"path": "/tmp/out.md", "body": "..."}, 70, 220, 500),
        ("search", {"q": "agent runtime architecture overview"}, 320, 240, 600),
        ("fetch_url", {"url": "https://a.example"}, 800, 200, 400),
        ("send_email", {"to": "team@example.com"}, 110, 200, 600),
        ("read_file", {"path": "/tmp/notes.md"}, 60, 160, 500),
        ("search", {"q": "agent runtime architecture"}, 320, 220, 600),
        ("fetch_url", {"url": "https://a.example"}, 800, 200, 400),
        ("read_file", {"path": "/tmp/notes.md"}, 60, 160, 500),
    ]
    return [
        TrajectoryStep(
            call_id=f"c{i:03d}",
            tool_name=tool_name,
            args=args,
            exec_ms=exec_ms,
            stream_ms=stream_ms,
            think_ms=think_ms,
        )
        for i, (tool_name, args, exec_ms, stream_ms, think_ms) in enumerate(raw)
    ]


async def sleep_ms(ms: int) -> None:
    await asyncio.sleep(ms / 1000)


async def run_trajectory(
    trajectory: list[TrajectoryStep],
    *,
    with_precog: bool,
    with_speculative_execution: bool,
) -> dict[str, Any]:
    tool_executions = 0
    cache_hits = 0

    async def executor(tool_name: str, args: Mapping[str, Any]) -> dict[str, Any]:
        found = next(
            (
                step
                for step in trajectory
                if step.tool_name == tool_name and dict(step.args) == dict(args)
            ),
            None,
        )
        await sleep_ms(found.exec_ms if found else 200)
        return {"tool": tool_name, "args": dict(args)}

    precog = None
    if with_precog:
        precog = PreCog(
            read_only_tools=READ_ONLY_TOOLS,
            executor=executor if with_speculative_execution else None,
        )

    started = time.monotonic()
    for step in trajectory:
        if precog is not None:
            await precog.observe_model_event(
                {
                    "type": "tool_call_start",
                    "callId": step.call_id,
                    "toolName": step.tool_name,
                }
            )
            await precog.observe_model_event(
                {
                    "type": "tool_call_args_delta",
                    "callId": step.call_id,
                    "delta": json.dumps(step.args),
                }
            )

        await sleep_ms(step.stream_ms)

        if precog is not None:
            await precog.observe_model_event(
                {"type": "tool_call_end", "callId": step.call_id}
            )

        await sleep_ms(step.think_ms)

        decision = None
        if precog is not None:
            decision = await precog.before_execute(
                step.tool_name,
                step.args,
                call_id=step.call_id,
            )

        if decision is not None and decision.type == "provide_result":
            cache_hits += 1
            continue

        await sleep_ms(step.exec_ms)
        tool_executions += 1
        if precog is not None:
            await precog.after_execute(
                step.tool_name,
                step.args,
                {"tool": step.tool_name, "args": dict(step.args)},
                call_id=step.call_id,
            )

    total_ms = int((time.monotonic() - started) * 1000)
    return {
        "total_ms": total_ms,
        "tool_executions": tool_executions,
        "cache_hits": cache_hits,
        "config": precog.snapshot() if precog is not None else None,
    }


def fmt(label: str, value: int) -> str:
    return f"{label}={value:>5}"


async def main() -> None:
    trajectory = build_trajectory()
    raw_total = sum(s.stream_ms + s.think_ms + s.exec_ms for s in trajectory)
    print(
        f"\nBenchmark: {len(trajectory)} tool calls, theoretical sequential "
        f"wall time = {raw_total}ms.\n"
    )

    baseline = await run_trajectory(
        trajectory,
        with_precog=False,
        with_speculative_execution=False,
    )
    print("[1/3] baseline")
    print(
        f"  {fmt('total', baseline['total_ms'])}  "
        f"{fmt('exec', baseline['tool_executions'])}  "
        f"{fmt('hits', baseline['cache_hits'])}"
    )

    memo = await run_trajectory(
        trajectory,
        with_precog=True,
        with_speculative_execution=False,
    )
    print("\n[2/3] PreCog memoization-only")
    print(
        f"  {fmt('total', memo['total_ms'])}  "
        f"{fmt('exec', memo['tool_executions'])}  "
        f"{fmt('hits', memo['cache_hits'])}"
    )

    spec = await run_trajectory(
        trajectory,
        with_precog=True,
        with_speculative_execution=True,
    )
    print("\n[3/3] PreCog memoization + speculative execution")
    print(
        f"  {fmt('total', spec['total_ms'])}  "
        f"{fmt('exec', spec['tool_executions'])}  "
        f"{fmt('hits', spec['cache_hits'])}"
    )

    memo_saved = baseline["total_ms"] - memo["total_ms"]
    spec_saved = baseline["total_ms"] - spec["total_ms"]
    print("\nResults vs baseline:")
    print(f"  memoization-only      -{memo_saved:>4}ms")
    print(f"  memo + spec exec      -{spec_saved:>4}ms")
    print("\nPreCog stats:")
    print(json.dumps(spec["config"], indent=2))


if __name__ == "__main__":
    asyncio.run(main())

