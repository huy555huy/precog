# PreCog

[简体中文](README.zh-CN.md)

Python-first speculative tool execution for agent runtimes.

PreCog watches streamed model tool-call events. Once it sees a read-only tool
call and its streamed arguments, it can pre-execute the tool while the model is
still producing the rest of the turn. When the real tool hook arrives, PreCog
returns a cached result if the actual `(tool, args)` pair matches.

It is the same latency trick used by CPUs and speculative decoders, moved to
the agent tool layer:

| Layer | Speculative trick | Wasted on miss | Saved on hit |
| --- | --- | --- | --- |
| CPU | branch prediction | flushed pipeline | throughput |
| LLM runtime | draft tokens | discarded tokens | decoding latency |
| PreCog | pre-run safe tool calls | wasted read-only call | agent wall time |

## Status

Alpha runtime. This repo is independent Python code with no Node or pnpm
dependency.

Implemented:

- idempotency registry for read-only tool allowlisting
- stable JSON cache keys for exact argument matching
- TTL/LRU speculation cache
- token-Jaccard fuzzy fallback
- bigram predictor with recent-argument memory
- tool-name-stage speculation with guessed recent args
- cancellation for wrong in-flight guesses
- adaptive pause when cache hit-rate drops too low
- max in-flight speculation throttling
- local `ToolRegistry` for real Python functions
- optional OpenAI Responses and LangGraph adapters
- JSONL tracing and predictor state persistence
- richer runtime stats for resolved, cancelled, wasted, and paused work
- async speculative executor
- `before_execute` / `after_execute` integration hooks
- smoke demo, synthetic benchmark, and unittest suite

## Install

```bash
python -m pip install -e .
```

For local development without installing:

```bash
PYTHONPATH=src python -m unittest discover -s tests
PYTHONPATH=src python demo/smoke.py
PYTHONPATH=src python demo/registry_quickstart.py
PYTHONPATH=src python demo/benchmark.py
```

## Quickstart

```python
import asyncio
import json
from typing import Any, Mapping

from precog import PreCog


async def tool_executor(tool_name: str, args: Mapping[str, Any]) -> Any:
    # Call your real tool runner here.
    await asyncio.sleep(0.2)
    return {"tool": tool_name, "args": dict(args)}


async def main() -> None:
    precog = PreCog(
        read_only_tools={"search", "fetch_url", "read_file"},
        executor=tool_executor,
    )

    call_id = "call-1"
    args = {"q": "agent runtime architecture"}

    await precog.observe_model_event(
        {"type": "tool_call_start", "callId": call_id, "toolName": "search"}
    )
    await precog.observe_model_event(
        {"type": "tool_call_args_delta", "callId": call_id, "delta": json.dumps(args)}
    )
    await precog.observe_model_event({"type": "tool_call_end", "callId": call_id})

    decision = await precog.before_execute("search", args, call_id=call_id)
    if decision.type == "provide_result":
        result = decision.result
    else:
        result = await tool_executor("search", args)
        await precog.after_execute("search", args, result, call_id=call_id)

    print(result)


asyncio.run(main())
```

If your integration does not expose separate before/after hooks, use the
convenience wrapper:

```python
result = await precog.execute("search", {"q": "python agents"}, tool_executor)
```

## Real Tool Registry

For a standalone Python app, use `ToolRegistry` as the dispatcher:

```python
from precog import IdempotencyClass, PreCog, ToolRegistry

registry = ToolRegistry()


@registry.register(idempotency_class=IdempotencyClass.NETWORK_READ, timeout_seconds=3)
def search(q: str) -> dict[str, str]:
    return {"query": q}


precog = PreCog(**registry.precog_kwargs())
result = await precog.execute("search", {"q": "agent latency"}, registry.execute)
```

Sync tools are offloaded to a worker thread by default so speculative execution
does not block the event loop.

## Runtime Controls

The default settings are aggressive enough to show latency wins, but still keep
guardrails on:

```python
precog = PreCog(
    read_only_tools={"search", "fetch_url"},
    executor=tool_executor,
    predict_on_tool_start=True,
    adaptive_min_calls=20,
    adaptive_min_hit_rate=0.2,
    adaptive_cooldown_seconds=30,
    max_concurrent_speculations=8,
)
```

- `predict_on_tool_start` starts speculation as soon as the model names a tool,
  using that tool's most recently observed arguments.
- if streamed arguments later disagree with the guessed args, the wrong
  in-flight task is cancelled and counted as wasted speculation.
- the adaptive guard pauses new speculation temporarily when the observed
  hit-rate falls below `adaptive_min_hit_rate`.
- `max_concurrent_speculations` prevents speculation from stealing unlimited
  resources from authoritative tool calls.

## Integrations

### OpenAI Responses streaming

```python
from precog.adapters.openai import OpenAIResponsesAdapter

adapter = OpenAIResponsesAdapter()

for event in stream:
    for model_event in adapter.events_from(event):
        await precog.observe_model_event(model_event)
```

The adapter translates function-call argument deltas into PreCog's generic
`ModelEvent` shape.

### LangGraph ToolNode

```python
from langgraph.prebuilt import ToolNode
from precog.adapters.langgraph import make_langgraph_tool_wrappers

tool_node = ToolNode(
    tools,
    **make_langgraph_tool_wrappers(precog),
)
```

The wrappers call `before_execute`; on a hit they return the cached ToolNode
result without invoking the tool, otherwise they execute normally and memoize in
`after_execute`.

## Operations

```python
from precog import JsonlTraceSink

precog = PreCog(
    **registry.precog_kwargs(),
    trace_sink=JsonlTraceSink("logs/precog.jsonl"),
)

precog.save_state("precog-state.json")
precog.load_state("precog-state.json")
await precog.close(cancel=True)
```

See [research notes](docs/research-notes.md) for the official framework hooks
and papers that shaped this version.

## Safety Model

Speculative execution is opt-in. Tools are treated as unsafe unless you mark
them read-only:

```python
precog = PreCog(read_only_tools={"search", "fetch_url"})
```

You can also classify tools explicitly:

```python
precog = PreCog(
    idempotency_classes={
        "search": "network_read",
        "write_file": "local_write_idempotent",
        "send_email": "remote_write",
    }
)
```

Only `pure_read` and `network_read` are allowed to run speculatively. Mutating
tools will miss the cache and execute normally.

## Benchmark

The demo benchmark models 12 tool calls with streamed model time, post-stream
thinking time, and real tool latency:

```bash
PYTHONPATH=src python demo/benchmark.py
```

It compares:

1. baseline with no PreCog
2. memoization-only cache hits
3. memoization plus in-stream speculative execution

The exact numbers depend on your machine and event loop timing, but the shape
should show fewer real tool executions and lower wall time once read-only calls
repeat or can be pre-run.

## GitHub

This project is ready to push as a standalone Python repo:

```bash
git add .
git commit -m "Rewrite PreCog as a Python package"
git branch -M main
git remote add origin git@github.com:<you>/precog.git
git push -u origin main
```

Replace `<you>` with your GitHub username or organization.

## License

MIT
