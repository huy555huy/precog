from __future__ import annotations

import asyncio
import json
import time
import unittest
from typing import Any, Mapping

from precog import PreCog, SpeculationCache, cache_key, jaccard, tokenize


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


if __name__ == "__main__":
    unittest.main()
