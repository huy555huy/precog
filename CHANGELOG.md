# Changelog

## 0.7.0

- Add a stdlib Anthropic streaming client that accumulates SSE events into a
  final Messages response while forwarding events to PreCog.
- Add a live `examples/agent_bench.py` harness with pass/fail scoring,
  tool-call counts, cache hits, shadow hits, speculation counts, and Markdown
  reports.
- Document a real relay-backed agent benchmark in the README.
- Harden streaming tool-use handling for relays that omit `stop_reason`.

## 0.6.0

- Add Anthropic/Claude Messages adapter for `tool_use` / `tool_result` loops.
- Add Anthropic streaming event translation for `input_json_delta` tool args.
- Add `ToolRegistry.anthropic_tools()` schema generation.
- Add Anthropic eval harness with baseline/observe/memoize/speculate modes.
- Add Anthropic Messages demo and tests for tool-call extraction and results.

## 0.5.0

- Add rollout modes: `off`, `observe`, `memoize`, and `speculate` for safer
  staged adoption.
- Add shadow-hit accounting for observe-mode rollouts.
- Add Prometheus-style metrics export through `PreCog.metrics_text()`.
- Persist JSON-serializable cache entries in state files with remaining TTL.
- Add `python -m precog` CLI commands: `doctor`, `metrics`, and `inspect-state`.
- Add next-tool confidence threshold support for cross-turn speculation.
- Expand tests to cover rollout modes, state/cache restore, metrics, and CLI.

## 0.4.0

- Add OpenAI Responses helpers to extract function calls, execute them through
  PreCog, and emit `function_call_output` input items.
- Add OpenAI-compatible tool schema generation from `ToolRegistry` function
  signatures and docstrings.
- Add LangChain callback observer for predictor training and memoization.
- Expand tests for OpenAI response execution, function-call output formatting,
  LangChain callback observation, and generated tool schemas.

## 0.3.0

- Add `ToolRegistry` for real local Python tool execution with idempotency classes,
  sync/async support, thread offloading, and per-tool timeouts.
- Add optional LangGraph `ToolNode` wrappers for cache short-circuiting via
  `wrap_tool_call` / `awrap_tool_call`.
- Add OpenAI Responses streaming event adapter for function-call deltas.
- Add JSONL tracing, predictor state save/load, graceful drain/close lifecycle,
  and max in-flight speculation throttling.
- Add research notes that tie the implementation to official framework hooks and
  speculative-tool-execution papers.
- Expand the test suite to cover registry execution, adapters, tracing,
  persistence, and throttling.

## 0.2.0

- Start speculation earlier at `tool_call_start` when recent args are available.
- Cancel in-flight tool-start guesses when streamed args prove them wrong.
- Add adaptive speculation pause for low observed hit-rate.
- Expand stats with tool-start launches, cancellations, wasted work, and pauses.
- Add tests for early speculation, cancellation, and adaptive pause behavior.

## 0.1.0

- Initial Python package rewrite.
- Add idempotency registry, speculation cache, predictor, runtime hooks, demos,
  and unit tests.
