from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Mapping
from uuid import uuid4

from precog import (
    IdempotencyClass,
    JsonlTraceSink,
    PreCog,
    SpeculationCache,
    ToolRegistry,
    cache_key,
    jaccard,
    tokenize,
)
from precog.adapters.langgraph import make_langgraph_tool_wrappers
from precog.adapters.langchain import PreCogLangChainCallbackHandler
from precog.adapters.anthropic import (
    AnthropicMessagesAdapter,
    execute_message_tool_calls,
    extract_tool_uses,
    tool_result_block,
)
from precog.adapters.openai import (
    OpenAIResponsesAdapter,
    execute_response_tool_calls,
    extract_function_calls,
    function_call_output,
)
from precog.__main__ import main as precog_cli


class CacheTests(unittest.TestCase):
    def test_cache_key_is_stable_across_argument_order(self) -> None:
        self.assertEqual(
            cache_key("search", {"b": 2, "a": 1}),
            cache_key("search", {"a": 1, "b": 2}),
        )

    def test_fuzzy_lookup_uses_token_jaccard(self) -> None:
        cache = SpeculationCache()
        cache.set(
            "search",
            {"q": "agent runtime architecture"},
            {"ok": True},
            observed_latency_ms=123,
        )
        found = cache.fuzzy_find(
            "search",
            {"q": "agent runtime architecture overview"},
            threshold=0.7,
        )
        self.assertIsNotNone(found)
        assert found is not None
        entry, similarity = found
        self.assertEqual(entry.result, {"ok": True})
        self.assertGreaterEqual(similarity, 0.7)

    def test_token_jaccard(self) -> None:
        self.assertEqual(tokenize({"q": "Agent-runtime"}), {"agent", "runtime"})
        self.assertAlmostEqual(jaccard({"a", "b"}, {"b", "c"}), 1 / 3)


class PreCogTests(unittest.IsolatedAsyncioTestCase):
    async def test_after_execute_memoizes_read_only_tools(self) -> None:
        precog = PreCog(read_only_tools={"search"})
        first = await precog.before_execute("search", {"q": "x"}, call_id="c1")
        self.assertEqual(first.type, "allow")
        await precog.after_execute(
            "search",
            {"q": "x"},
            {"result": "x"},
            call_id="c1",
        )

        second = await precog.before_execute("search", {"q": "x"}, call_id="c2")
        self.assertEqual(second.type, "provide_result")
        self.assertEqual(second.result, {"result": "x"})
        self.assertEqual(precog.config().stats.strict_hits, 1)

    async def test_side_effecting_tools_are_not_cached(self) -> None:
        precog = PreCog(read_only_tools={"search"})
        await precog.before_execute("write_file", {"path": "x"}, call_id="c1")
        await precog.after_execute(
            "write_file",
            {"path": "x"},
            {"ok": True},
            call_id="c1",
        )
        second = await precog.before_execute("write_file", {"path": "x"}, call_id="c2")
        self.assertEqual(second.type, "allow")
        self.assertEqual(precog.config().cache_size, 0)

    async def test_streamed_args_launch_speculation(self) -> None:
        calls: list[tuple[str, Mapping[str, Any]]] = []

        async def executor(tool_name: str, args: Mapping[str, Any]) -> dict[str, Any]:
            calls.append((tool_name, dict(args)))
            await asyncio.sleep(0.01)
            return {"from": "spec", "tool": tool_name, "args": dict(args)}

        precog = PreCog(read_only_tools={"search"}, executor=executor)
        args = {"q": "agent runtime"}
        await precog.observe_model_event(
            {"type": "tool_call_start", "callId": "c1", "toolName": "search"}
        )
        await precog.observe_model_event(
            {
                "type": "tool_call_args_delta",
                "callId": "c1",
                "delta": json.dumps(args),
            }
        )
        await precog.observe_model_event({"type": "tool_call_end", "callId": "c1"})

        decision = await precog.before_execute("search", args, call_id="c1")

        self.assertEqual(decision.type, "provide_result")
        self.assertEqual(decision.result["from"], "spec")
        self.assertEqual(calls, [("search", args)])
        self.assertEqual(precog.config().stats.speculations_launched, 1)
        self.assertEqual(precog.config().stats.speculations_resolved, 1)

    async def test_tool_start_speculation_uses_recent_args(self) -> None:
        calls: list[tuple[str, Mapping[str, Any]]] = []

        async def executor(tool_name: str, args: Mapping[str, Any]) -> dict[str, Any]:
            calls.append((tool_name, dict(args)))
            await asyncio.sleep(0.01)
            return {"from": "tool_start", "tool": tool_name, "args": dict(args)}

        precog = PreCog(read_only_tools={"search"}, executor=executor)
        args = {"q": "agent runtime"}

        await precog.before_execute("search", args, call_id="seed")
        await precog.after_execute("search", args, {"seed": True}, call_id="seed")
        precog.cache.clear()

        await precog.observe_model_event(
            {"type": "tool_call_start", "callId": "c1", "toolName": "search"}
        )
        await asyncio.sleep(0)
        await precog.observe_model_event(
            {
                "type": "tool_call_args_delta",
                "callId": "c1",
                "delta": json.dumps(args),
            }
        )
        await precog.observe_model_event({"type": "tool_call_end", "callId": "c1"})

        decision = await precog.before_execute("search", args, call_id="c1")

        self.assertEqual(decision.type, "provide_result")
        self.assertEqual(decision.result["from"], "tool_start")
        self.assertEqual(calls, [("search", args)])
        self.assertEqual(precog.config().stats.tool_start_speculations, 1)
        self.assertEqual(precog.config().stats.speculations_launched, 0)

    async def test_wrong_tool_start_guess_is_cancelled(self) -> None:
        calls: list[tuple[str, Mapping[str, Any]]] = []
        cancelled: list[Mapping[str, Any]] = []

        async def executor(tool_name: str, args: Mapping[str, Any]) -> dict[str, Any]:
            calls.append((tool_name, dict(args)))
            try:
                await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                cancelled.append(dict(args))
                raise
            return {"tool": tool_name, "args": dict(args)}

        precog = PreCog(read_only_tools={"search"}, executor=executor)
        old_args = {"q": "old"}
        new_args = {"q": "new"}

        await precog.before_execute("search", old_args, call_id="seed")
        await precog.after_execute("search", old_args, {"seed": True}, call_id="seed")
        precog.cache.clear()

        await precog.observe_model_event(
            {"type": "tool_call_start", "callId": "c1", "toolName": "search"}
        )
        await asyncio.sleep(0)
        await precog.observe_model_event(
            {
                "type": "tool_call_args_delta",
                "callId": "c1",
                "delta": json.dumps(new_args),
            }
        )
        await precog.observe_model_event({"type": "tool_call_end", "callId": "c1"})

        decision = await precog.before_execute("search", new_args, call_id="c1")

        self.assertEqual(decision.type, "provide_result")
        self.assertEqual(decision.result["args"], new_args)
        self.assertEqual(cancelled, [old_args])
        self.assertEqual(calls, [("search", old_args), ("search", new_args)])
        self.assertEqual(precog.config().stats.wasted_speculations, 1)
        self.assertEqual(precog.config().stats.speculations_cancelled, 1)

    async def test_adaptive_pause_stops_new_speculations_after_misses(self) -> None:
        calls: list[tuple[str, Mapping[str, Any]]] = []

        async def executor(tool_name: str, args: Mapping[str, Any]) -> dict[str, Any]:
            calls.append((tool_name, dict(args)))
            return {"tool": tool_name, "args": dict(args)}

        precog = PreCog(
            read_only_tools={"search"},
            executor=executor,
            adaptive_min_calls=2,
            adaptive_min_hit_rate=0.5,
            adaptive_cooldown_seconds=60,
        )

        await precog.before_execute("search", {"q": "a"}, call_id="c1")
        await precog.before_execute("search", {"q": "b"}, call_id="c2")
        await precog.observe_model_event(
            {"type": "tool_call_start", "callId": "c3", "toolName": "search"}
        )
        await precog.observe_model_event(
            {
                "type": "tool_call_args_delta",
                "callId": "c3",
                "delta": json.dumps({"q": "c"}),
            }
        )
        await precog.observe_model_event({"type": "tool_call_end", "callId": "c3"})

        config = precog.config()
        self.assertTrue(config.speculation_paused)
        self.assertEqual(config.stats.adaptive_pauses, 1)
        self.assertEqual(config.stats.speculations_launched, 0)
        self.assertEqual(config.stats.tool_start_speculations, 0)
        self.assertEqual(calls, [])

    async def test_execute_wrapper_runs_real_tool_then_hits_cache(self) -> None:
        executions = 0

        async def runner(tool_name: str, args: Mapping[str, Any]) -> dict[str, Any]:
            nonlocal executions
            executions += 1
            await asyncio.sleep(0)
            return {"tool": tool_name, "args": dict(args), "at": time.time()}

        precog = PreCog(read_only_tools={"search"})
        one = await precog.execute("search", {"q": "x"}, runner, call_id="c1")
        two = await precog.execute("search", {"q": "x"}, runner, call_id="c2")

        self.assertEqual(executions, 1)
        self.assertEqual(one, two)

    async def test_observe_mode_records_shadow_hits_without_short_circuiting(self) -> None:
        executions = 0

        async def runner(tool_name: str, args: Mapping[str, Any]) -> dict[str, Any]:
            nonlocal executions
            executions += 1
            return {"tool": tool_name, "args": dict(args), "execution": executions}

        precog = PreCog(read_only_tools={"search"}, rollout_mode="observe")
        one = await precog.execute("search", {"q": "x"}, runner, call_id="c1")
        two = await precog.execute("search", {"q": "x"}, runner, call_id="c2")

        self.assertEqual(executions, 2)
        self.assertNotEqual(one, two)
        self.assertEqual(precog.config().stats.shadow_hits, 1)
        self.assertEqual(precog.config().stats.cache_hits, 0)

    async def test_memoize_mode_uses_cache_without_launching_speculation(self) -> None:
        async def executor(tool_name: str, args: Mapping[str, Any]) -> dict[str, Any]:
            return {"from": "spec", "tool": tool_name, "args": dict(args)}

        precog = PreCog(
            read_only_tools={"search"},
            executor=executor,
            rollout_mode="memoize",
        )
        args = {"q": "x"}
        await precog.observe_model_event(
            {"type": "tool_call_start", "callId": "c1", "toolName": "search"}
        )
        await precog.observe_model_event(
            {"type": "tool_call_args_delta", "callId": "c1", "delta": json.dumps(args)}
        )
        await precog.observe_model_event({"type": "tool_call_end", "callId": "c1"})

        await precog.before_execute("search", args, call_id="c1")
        await precog.after_execute("search", args, {"real": True}, call_id="c1")
        decision = await precog.before_execute("search", args, call_id="c2")

        self.assertEqual(decision.type, "provide_result")
        self.assertEqual(decision.result, {"real": True})
        self.assertEqual(precog.config().stats.speculations_launched, 0)

    async def test_off_mode_leaves_calls_untouched(self) -> None:
        precog = PreCog(read_only_tools={"search"}, rollout_mode="off")
        decision = await precog.before_execute("search", {"q": "x"}, call_id="c1")
        await precog.after_execute("search", {"q": "x"}, {"result": "x"}, call_id="c1")

        self.assertEqual(decision.type, "allow")
        self.assertEqual(precog.config().cache_size, 0)
        self.assertEqual(precog.config().stats.cache_misses, 0)

    async def test_tool_registry_executes_registered_tools(self) -> None:
        registry = ToolRegistry()

        @registry.register(idempotency_class=IdempotencyClass.NETWORK_READ)
        def search(q: str) -> dict[str, str]:
            """Search indexed docs."""
            return {"q": q}

        precog = PreCog(**registry.precog_kwargs())
        result = await precog.execute("search", {"q": "x"}, registry.execute)

        self.assertEqual(result, {"q": "x"})
        self.assertEqual(registry.read_only_tools(), ("search",))
        self.assertEqual(search("y"), {"q": "y"})
        self.assertEqual(
            registry.openai_tools(),
            [
                {
                    "type": "function",
                    "name": "search",
                    "description": "Search indexed docs.",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                        "required": ["q"],
                        "additionalProperties": False,
                    },
                }
            ],
        )
        self.assertEqual(
            registry.anthropic_tools(),
            [
                {
                    "name": "search",
                    "description": "Search indexed docs.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                        "required": ["q"],
                        "additionalProperties": False,
                    },
                }
            ],
        )

    async def test_predictor_state_can_round_trip_to_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "precog-state.json"
            one = PreCog(read_only_tools={"search"})
            await one.before_execute("search", {"q": "persist"}, call_id="c1")
            await one.after_execute("search", {"q": "persist"}, {"ok": True}, call_id="c1")
            one.save_state(path)

            two = PreCog(read_only_tools={"search"})
            two.load_state(path)

            self.assertEqual(two.predictor.guess_args("search"), {"q": "persist"})

    async def test_state_round_trip_persists_json_cache_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "precog-state.json"
            one = PreCog(read_only_tools={"search"})
            await one.before_execute("search", {"q": "persist"}, call_id="c1")
            await one.after_execute("search", {"q": "persist"}, {"ok": True}, call_id="c1")
            one.save_state(path)

            two = PreCog(read_only_tools={"search"})
            two.load_state(path)
            decision = await two.before_execute("search", {"q": "persist"}, call_id="c2")

            self.assertEqual(decision.type, "provide_result")
            self.assertEqual(decision.result, {"ok": True})

    async def test_metrics_text_exports_prometheus_style_stats(self) -> None:
        precog = PreCog(read_only_tools={"search"})
        await precog.before_execute("search", {"q": "x"}, call_id="c1")

        metrics = precog.metrics_text()

        self.assertIn("# TYPE precog_cache_misses counter", metrics)
        self.assertIn("precog_cache_misses 1.0", metrics)
        self.assertIn("precog_hit_rate", metrics)

    async def test_max_concurrent_speculations_throttles_launches(self) -> None:
        async def executor(tool_name: str, args: Mapping[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(0.02)
            return {"tool": tool_name, "args": dict(args)}

        precog = PreCog(
            read_only_tools={"search"},
            executor=executor,
            max_concurrent_speculations=1,
        )
        await precog.observe_model_event(
            {"type": "tool_call_start", "callId": "c1", "toolName": "search"}
        )
        await precog.observe_model_event(
            {"type": "tool_call_args_delta", "callId": "c1", "delta": json.dumps({"q": "a"})}
        )
        await precog.observe_model_event({"type": "tool_call_end", "callId": "c1"})
        await precog.observe_model_event(
            {"type": "tool_call_start", "callId": "c2", "toolName": "search"}
        )
        await precog.observe_model_event(
            {"type": "tool_call_args_delta", "callId": "c2", "delta": json.dumps({"q": "b"})}
        )
        await precog.observe_model_event({"type": "tool_call_end", "callId": "c2"})

        await precog.drain()

        self.assertEqual(precog.config().stats.speculations_launched, 1)
        self.assertEqual(precog.config().stats.speculations_throttled, 1)

    async def test_jsonl_trace_sink_records_speculation_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"

            async def executor(tool_name: str, args: Mapping[str, Any]) -> dict[str, Any]:
                return {"tool": tool_name, "args": dict(args)}

            precog = PreCog(
                read_only_tools={"search"},
                executor=executor,
                trace_sink=JsonlTraceSink(path),
            )
            await precog.observe_model_event(
                {"type": "tool_call_start", "callId": "c1", "toolName": "search"}
            )
            await precog.observe_model_event(
                {
                    "type": "tool_call_args_delta",
                    "callId": "c1",
                    "delta": json.dumps({"q": "x"}),
                }
            )
            await precog.observe_model_event({"type": "tool_call_end", "callId": "c1"})
            await precog.drain()

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertTrue(any('"event": "speculation_launched"' in line for line in lines))
            self.assertTrue(any('"event": "speculation_resolved"' in line for line in lines))


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_openai_responses_adapter_translates_function_events(self) -> None:
        adapter = OpenAIResponsesAdapter()
        started = adapter.events_from(
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "search",
                },
            }
        )
        delta = adapter.events_from(
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "delta": '{"q":',
            }
        )
        done = adapter.events_from(
            {
                "type": "response.function_call_arguments.done",
                "item_id": "fc_1",
                "arguments": '"x"}',
            }
        )

        events = started + delta + done
        self.assertEqual(events[0].type, "tool_call_start")
        self.assertEqual(events[0].call_id, "call_1")
        self.assertEqual(events[0].tool_name, "search")
        self.assertEqual(events[1].delta, '{"q":')
        self.assertEqual(events[2].delta, '"x"}')
        self.assertEqual(events[3].type, "tool_call_end")

    async def test_anthropic_messages_adapter_translates_streaming_tool_events(
        self,
    ) -> None:
        adapter = AnthropicMessagesAdapter()
        started = adapter.events_from(
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "search",
                    "input": {},
                },
            }
        )
        delta = adapter.events_from(
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"q":"x"}'},
            }
        )
        stopped = adapter.events_from({"type": "content_block_stop", "index": 1})

        events = started + delta + stopped
        self.assertEqual(events[0].type, "tool_call_start")
        self.assertEqual(events[0].call_id, "toolu_1")
        self.assertEqual(events[0].tool_name, "search")
        self.assertEqual(events[1].delta, '{"q":"x"}')
        self.assertEqual(events[2].type, "tool_call_end")

    async def test_anthropic_tool_call_helper_executes_and_formats_results(self) -> None:
        message = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "search",
                    "input": {"q": "x"},
                }
            ]
        }

        async def runner(tool_name: str, args: Mapping[str, Any]) -> dict[str, Any]:
            return {"tool": tool_name, "args": dict(args)}

        precog = PreCog(read_only_tools={"search"})
        uses = extract_tool_uses(message)
        blocks = await execute_message_tool_calls(precog, message, runner)

        self.assertEqual(uses[0].name, "search")
        self.assertEqual(uses[0].input, {"q": "x"})
        self.assertEqual(
            blocks,
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": '{"tool": "search", "args": {"q": "x"}}',
                }
            ],
        )

    async def test_anthropic_tool_result_block_preserves_string_outputs(self) -> None:
        self.assertEqual(
            tool_result_block("toolu_1", "plain"),
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "plain"},
        )

    async def test_openai_response_tool_call_helper_executes_and_formats_outputs(
        self,
    ) -> None:
        response = {
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "search",
                    "arguments": '{"q": "x"}',
                }
            ]
        }

        async def runner(tool_name: str, args: Mapping[str, Any]) -> dict[str, Any]:
            return {"tool": tool_name, "args": dict(args)}

        precog = PreCog(read_only_tools={"search"})
        calls = extract_function_calls(response)
        outputs = await execute_response_tool_calls(precog, response, runner)

        self.assertEqual(calls[0].name, "search")
        self.assertEqual(calls[0].arguments, {"q": "x"})
        self.assertEqual(
            outputs,
            [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": '{"tool": "search", "args": {"q": "x"}}',
                }
            ],
        )

    async def test_openai_function_call_output_preserves_string_outputs(self) -> None:
        self.assertEqual(
            function_call_output("call_1", "plain"),
            {"type": "function_call_output", "call_id": "call_1", "output": "plain"},
        )

    async def test_langgraph_async_wrapper_caches_tool_messages(self) -> None:
        class Request:
            tool_call = {"name": "search", "args": {"q": "x"}, "id": "call-1"}

        executions = 0

        async def execute(request: Request) -> dict[str, Any]:
            nonlocal executions
            executions += 1
            return {"content": "fresh", "tool_call_id": request.tool_call["id"]}

        precog = PreCog(read_only_tools={"search"})
        wrapper = make_langgraph_tool_wrappers(precog)["awrap_tool_call"]

        one = await wrapper(Request(), execute)
        two = await wrapper(Request(), execute)

        self.assertEqual(executions, 1)
        self.assertEqual(one, two)

    async def test_langchain_callback_observes_tool_results_without_cache_hit_stats(
        self,
    ) -> None:
        precog = PreCog(read_only_tools={"search"})
        handler = PreCogLangChainCallbackHandler(precog)
        run_id = uuid4()

        await handler.on_tool_start(
            {"name": "search"},
            '{"q": "x"}',
            run_id=run_id,
        )
        await handler.on_tool_end({"result": "x"}, run_id=run_id)

        decision = await precog.before_execute("search", {"q": "x"}, call_id="c2")

        self.assertEqual(decision.type, "provide_result")
        self.assertEqual(decision.result, {"result": "x"})
        self.assertEqual(precog.config().stats.cache_hits, 1)
        self.assertEqual(precog.config().stats.cache_misses, 0)


class CliTests(unittest.TestCase):
    def test_doctor_cli_prints_json(self) -> None:
        out = StringIO()
        with redirect_stdout(out):
            code = precog_cli(["doctor"])

        payload = json.loads(out.getvalue())
        self.assertEqual(code, 0)
        self.assertIn("python", payload)
        self.assertIn("precog", payload)

    def test_metrics_cli_prints_prometheus_text(self) -> None:
        out = StringIO()
        with redirect_stdout(out):
            code = precog_cli(["metrics"])

        self.assertEqual(code, 0)
        self.assertIn("precog_hit_rate", out.getvalue())

    def test_inspect_state_cli_summarizes_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            path.write_text(
                json.dumps(
                    {
                        "predictor": {
                            "args_memory": {"search": []},
                            "bigrams": {"search": {"fetch": 1}},
                        },
                        "cache": {"entries": [{"tool_name": "search"}]},
                    }
                ),
                encoding="utf-8",
            )
            out = StringIO()
            with redirect_stdout(out):
                code = precog_cli(["inspect-state", str(path)])

        payload = json.loads(out.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(payload["args_memory_tools"], 1)
        self.assertEqual(payload["bigram_sources"], 1)
        self.assertEqual(payload["cache_entries"], 1)


if __name__ == "__main__":
    unittest.main()
