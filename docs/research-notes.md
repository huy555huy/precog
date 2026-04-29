# Research Notes

These notes summarize the implementation direction for making PreCog practical
instead of just a benchmark toy.

## What The Current Ecosystem Exposes

- OpenAI function calling returns function-call items with a `call_id`, `name`,
  and JSON-encoded `arguments`; applications execute those calls and submit
  outputs back to the model. This maps directly to PreCog's `(tool, args)` cache
  key and `before_execute` / `after_execute` lifecycle.
  Source: [OpenAI Function calling](https://developers.openai.com/api/docs/guides/function-calling).
- OpenAI Responses streaming uses typed semantic events and includes function
  argument delta/done events. This gives PreCog an early signal before the
  final tool execution path.
  Source: [OpenAI Streaming API responses](https://developers.openai.com/api/docs/guides/streaming-responses).
- OpenAI Agents SDK exposes tool lifecycle hooks. For function tools, the hook
  context carries tool-call metadata such as id, name, and arguments.
  Source: [OpenAI Agents SDK lifecycle](https://openai.github.io/openai-agents-python/ref/lifecycle/).
- LangGraph's `ToolNode` accepts `wrap_tool_call` and `awrap_tool_call`
  interceptors, and its request object exposes the tool call dict. This is the
  cleanest production integration point for cache short-circuiting.
  Sources: [LangGraph ToolNode](https://reference.langchain.com/python/langgraph.prebuilt/tool_node/ToolNode),
  [ToolCallRequest](https://reference.langchain.com/python/langgraph.prebuilt/tool_node/ToolCallRequest),
  [ToolCallWrapper](https://reference.langchain.com/python/langgraph.prebuilt/tool_node/ToolCallWrapper).

## What The Research Suggests

- PASTE argues that agent tool flows have recurring control-flow patterns and
  predictable parameter dependencies; it reports 48.5% average completion-time
  reduction and 1.8x tool-throughput improvement. The important product lesson
  is that speculation should be pattern-aware and policy-gated, not a blind
  always-on firehose.
  Source: [arXiv:2603.18897](https://arxiv.org/abs/2603.18897).

## Product Requirements Derived From This

- Keep tool eligibility explicit. Mutating tools must be unsafe by default.
- Expose a real local tool registry so users can run PreCog without building
  their own dispatcher first.
- Integrate where frameworks already provide hooks instead of monkeypatching
  internals.
- Add resource controls: max in-flight speculation, cancellation, drain/close,
  and adaptive pause.
- Persist learned patterns across process restarts, but do not persist tool
  result cache by default because result freshness is application-specific.
- Emit traces that operators can inspect after a bad prediction or latency win.

## Implemented In 0.3.0

- `ToolRegistry` for local function execution and idempotency metadata.
- OpenAI Responses event adapter.
- LangGraph ToolNode wrapper helpers.
- JSONL tracing.
- Predictor state export/import/save/load.
- Max concurrent speculation throttling.
- `drain()` / `close()` lifecycle methods.
