from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import inspect
import json
from pathlib import Path
import time
from typing import Any, Awaitable, Callable, Literal, Mapping
from uuid import uuid4

from .cache import SpeculationCache, cache_key
from .idempotency import IdempotencyClass, IdempotencyRegistry
from .predictor import ToolCallPredictor
from .stats import PreCogStats, StatsCollector
from .trace import TraceSink


SpeculativeExecutor = Callable[[str, Mapping[str, Any]], Any | Awaitable[Any]]
RuntimeMode = Literal["off", "observe", "memoize", "speculate"]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class ModelEvent:
    type: str
    call_id: str
    tool_name: str | None = None
    delta: str | None = None


@dataclass(frozen=True)
class PreCogDecision:
    type: Literal["allow", "provide_result"]
    result: Any = None

    @property
    def should_execute(self) -> bool:
        return self.type == "allow"


@dataclass(frozen=True)
class PreCogConfig:
    id: str
    cache_ttl_seconds: float
    cache_max_entries: int
    read_only_tools: tuple[str, ...]
    rollout_mode: RuntimeMode
    speculative_execution: bool
    fuzzy_threshold: float
    min_next_tool_confidence: float
    predict_on_tool_start: bool
    adaptive_min_calls: int
    adaptive_min_hit_rate: float
    adaptive_cooldown_seconds: float
    max_concurrent_speculations: int
    speculation_paused: bool
    stats: PreCogStats
    hit_rate: float
    cache_size: int


@dataclass
class _ArgsBufferEntry:
    tool_name: str
    args_text: str = ""


class PreCog:
    """Python runtime for speculative tool execution.

    Integrate it around your agent's model-stream and tool-execution hooks:
    call ``observe_model_event`` while the model streams, ``before_execute``
    before a tool runs, and ``after_execute`` once a real tool call finishes.
    """

    def __init__(
        self,
        *,
        id: str = "precog",
        read_only_tools: set[str] | list[str] | tuple[str, ...] = (),
        idempotency_classes: Mapping[str, IdempotencyClass | str] | None = None,
        cache_max_entries: int = 64,
        cache_ttl_seconds: float = 60.0,
        history_window: int = 32,
        min_observations: int = 1,
        args_window: int = 8,
        fuzzy_threshold: float = 0.7,
        rollout_mode: RuntimeMode = "speculate",
        min_next_tool_confidence: float = 0.0,
        predict_on_tool_start: bool = True,
        adaptive_min_calls: int = 20,
        adaptive_min_hit_rate: float = 0.2,
        adaptive_cooldown_seconds: float = 30.0,
        max_concurrent_speculations: int = 8,
        executor: SpeculativeExecutor | None = None,
        trace_sink: TraceSink | None = None,
        verbose: bool = False,
    ) -> None:
        if not 0 <= fuzzy_threshold <= 1:
            raise ValueError("fuzzy_threshold must be between 0 and 1")
        if rollout_mode not in {"off", "observe", "memoize", "speculate"}:
            raise ValueError("rollout_mode must be off, observe, memoize, or speculate")
        if not 0 <= min_next_tool_confidence <= 1:
            raise ValueError("min_next_tool_confidence must be between 0 and 1")
        if adaptive_min_calls < 0:
            raise ValueError("adaptive_min_calls must be >= 0")
        if not 0 <= adaptive_min_hit_rate <= 1:
            raise ValueError("adaptive_min_hit_rate must be between 0 and 1")
        if adaptive_cooldown_seconds < 0:
            raise ValueError("adaptive_cooldown_seconds must be >= 0")
        if max_concurrent_speculations < 1:
            raise ValueError("max_concurrent_speculations must be >= 1")

        self.id = id
        self.read_only_tools = tuple(read_only_tools)
        self.idempotency = IdempotencyRegistry(
            read_only_tools=self.read_only_tools,
            classes=idempotency_classes,
        )
        self.cache = SpeculationCache(
            max_entries=cache_max_entries,
            ttl_seconds=cache_ttl_seconds,
        )
        self.predictor = ToolCallPredictor(
            history_window=history_window,
            min_observations=min_observations,
            args_window=args_window,
        )
        self.stats = StatsCollector()
        self.fuzzy_threshold = fuzzy_threshold
        self.rollout_mode = rollout_mode
        self.min_next_tool_confidence = min_next_tool_confidence
        self.predict_on_tool_start = predict_on_tool_start
        self.adaptive_min_calls = adaptive_min_calls
        self.adaptive_min_hit_rate = adaptive_min_hit_rate
        self.adaptive_cooldown_seconds = adaptive_cooldown_seconds
        self.max_concurrent_speculations = max_concurrent_speculations
        self.executor = executor
        self.trace_sink = trace_sink
        self.verbose = verbose

        self._args_buffer: dict[str, _ArgsBufferEntry] = {}
        self._inflight_speculations: dict[str, asyncio.Task[None]] = {}
        self._call_speculation_keys: dict[str, set[str]] = {}
        self._inflight_start: dict[str, float] = {}
        self._speculation_paused_until = 0.0

    async def observe_model_event(self, event: ModelEvent | Mapping[str, Any]) -> None:
        """Observe a streamed model event.

        Dict events may use either Python names (``call_id``/``tool_name``) or
        common JS hook names (``callId``/``toolName``).
        """

        if self.rollout_mode == "off":
            return

        event_type = _event_value(event, "type")
        call_id = _event_value(event, "call_id", "callId")
        if not event_type or not call_id:
            return

        if event_type == "tool_call_start":
            tool_name = _event_value(event, "tool_name", "toolName")
            if not tool_name:
                return
            self.stats.bump("observed_tool_starts")
            self._args_buffer[call_id] = _ArgsBufferEntry(tool_name=tool_name)
            if self.predict_on_tool_start:
                guessed_args = self.predictor.guess_args(tool_name)
                if guessed_args is not None:
                    self._try_start_speculation(
                        tool_name,
                        guessed_args,
                        "tool_start",
                        call_id=call_id,
                    )
            return

        if event_type == "tool_call_args_delta":
            entry = self._args_buffer.get(call_id)
            if entry is not None:
                entry.args_text += str(_event_value(event, "delta", default=""))
            return

        if event_type == "tool_call_end":
            entry = self._args_buffer.pop(call_id, None)
            if entry is None:
                return
            try:
                args = json.loads(entry.args_text or "{}")
            except json.JSONDecodeError:
                self._log(
                    f"spec SKIP tool={entry.tool_name} reason=unparseable_args"
                )
                return
            if not isinstance(args, Mapping):
                self._log(f"spec SKIP tool={entry.tool_name} reason=args_not_object")
                return
            normalized_args = dict(args)
            self._cancel_wrong_call_speculations(
                call_id,
                cache_key(entry.tool_name, normalized_args),
            )
            self._try_start_speculation(
                entry.tool_name,
                normalized_args,
                "in_stream",
                call_id=call_id,
            )

    async def before_execute(
        self,
        tool_name: str,
        args: Mapping[str, Any],
        *,
        call_id: str | None = None,
    ) -> PreCogDecision:
        """Return cached speculative result when available, otherwise allow."""

        if self.rollout_mode == "off":
            return PreCogDecision(type="allow")

        call_id = call_id or str(uuid4())
        normalized_args = dict(args)
        key = cache_key(tool_name, normalized_args)

        task = self._inflight_speculations.get(key)
        if task is not None:
            self._log(f"cache WAIT call_id={call_id} tool={tool_name}")
            await task

        if self.idempotency.is_safe_for_speculation(tool_name):
            strict = self.cache.peek_key(key)
            if strict is not None:
                self.stats.bump("strict_hits")
                if self._can_provide_cached_result():
                    self.stats.bump("cache_hits")
                    self.stats.bump("latency_saved_ms", strict.observed_latency_ms)
                else:
                    self.stats.bump("shadow_hits")
                self._log(
                    f"cache HIT strict call_id={call_id} tool={tool_name} "
                    f"saved~={strict.observed_latency_ms:.0f}ms"
                )
                self._maybe_pause_speculation()
                self._observe_and_cross_predict(tool_name, normalized_args)
                if self._can_provide_cached_result():
                    return PreCogDecision(type="provide_result", result=strict.result)
                self._inflight_start[call_id] = time.monotonic()
                return PreCogDecision(type="allow")

            if self.fuzzy_threshold < 1:
                fuzzy = self.cache.fuzzy_find(
                    tool_name,
                    normalized_args,
                    self.fuzzy_threshold,
                )
                if fuzzy is not None:
                    entry, similarity = fuzzy
                    self.stats.bump("fuzzy_hits")
                    if self._can_provide_cached_result():
                        self.stats.bump("cache_hits")
                        self.stats.bump("latency_saved_ms", entry.observed_latency_ms)
                    else:
                        self.stats.bump("shadow_hits")
                    self._log(
                        f"cache HIT fuzzy={similarity:.2f} call_id={call_id} "
                        f"tool={tool_name} saved~={entry.observed_latency_ms:.0f}ms"
                    )
                    self._maybe_pause_speculation()
                    self._observe_and_cross_predict(tool_name, normalized_args)
                    if self._can_provide_cached_result():
                        return PreCogDecision(type="provide_result", result=entry.result)
                    self._inflight_start[call_id] = time.monotonic()
                    return PreCogDecision(type="allow")

        self.stats.bump("cache_misses")
        self._inflight_start[call_id] = time.monotonic()
        self._maybe_pause_speculation()
        self._log(f"cache MISS call_id={call_id} tool={tool_name}")
        return PreCogDecision(type="allow")

    async def after_execute(
        self,
        tool_name: str,
        args: Mapping[str, Any],
        result: Any,
        *,
        call_id: str | None = None,
        is_error: bool = False,
    ) -> None:
        """Observe a completed real tool execution and memoize safe results."""

        if self.rollout_mode == "off":
            return

        normalized_args = dict(args)
        elapsed_ms = 0.0
        if call_id is not None:
            started_at = self._inflight_start.pop(call_id, None)
            if started_at is not None:
                elapsed_ms = (time.monotonic() - started_at) * 1000

        if not is_error and self.idempotency.is_safe_for_speculation(tool_name):
            self.cache.set(
                tool_name,
                normalized_args,
                result,
                observed_latency_ms=elapsed_ms,
            )
            self.stats.bump("cache_stores")
            self._log(
                f"cache STORE call_id={call_id or '-'} tool={tool_name} "
                f"latency={elapsed_ms:.0f}ms"
            )

        self._observe_and_cross_predict(tool_name, normalized_args)

    async def execute(
        self,
        tool_name: str,
        args: Mapping[str, Any],
        runner: SpeculativeExecutor,
        *,
        call_id: str | None = None,
    ) -> Any:
        """Convenience wrapper for direct integrations."""

        call_id = call_id or str(uuid4())
        decision = await self.before_execute(tool_name, args, call_id=call_id)
        if decision.type == "provide_result":
            return decision.result

        result = runner(tool_name, args)
        if inspect.isawaitable(result):
            result = await result
        await self.after_execute(tool_name, args, result, call_id=call_id)
        return result

    def config(self) -> PreCogConfig:
        return PreCogConfig(
            id=self.id,
            cache_ttl_seconds=self.cache.ttl_seconds,
            cache_max_entries=self.cache.max_entries,
            read_only_tools=self.idempotency.safe_tool_names(),
            rollout_mode=self.rollout_mode,
            speculative_execution=self.executor is not None,
            fuzzy_threshold=self.fuzzy_threshold,
            min_next_tool_confidence=self.min_next_tool_confidence,
            predict_on_tool_start=self.predict_on_tool_start,
            adaptive_min_calls=self.adaptive_min_calls,
            adaptive_min_hit_rate=self.adaptive_min_hit_rate,
            adaptive_cooldown_seconds=self.adaptive_cooldown_seconds,
            max_concurrent_speculations=self.max_concurrent_speculations,
            speculation_paused=self._is_speculation_paused(),
            stats=self.stats.snapshot(),
            hit_rate=self.stats.hit_rate(),
            cache_size=self.cache.size(),
        )

    def snapshot(self) -> dict[str, Any]:
        return asdict(self.config())

    def export_state(self) -> dict[str, Any]:
        return {"predictor": self.predictor.to_dict(), "cache": self.cache.to_dict()}

    def import_state(self, state: Mapping[str, Any]) -> None:
        predictor_state = state.get("predictor")
        if isinstance(predictor_state, Mapping):
            self.predictor.load_dict(predictor_state)
        cache_state = state.get("cache")
        if isinstance(cache_state, Mapping):
            self.cache.load_dict(cache_state)

    def save_state(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.export_state(), ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    def load_state(self, path: str | Path) -> None:
        self.import_state(json.loads(Path(path).read_text(encoding="utf-8")))

    def save_cache(self, path: str | Path) -> None:
        self.cache.save(path)

    def load_cache(self, path: str | Path) -> None:
        self.cache.load(path)

    def metrics_text(self, prefix: str = "precog") -> str:
        return self.stats.prometheus_text(prefix)

    async def drain(self) -> None:
        tasks = list(self._inflight_speculations.values())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self, *, cancel: bool = False) -> None:
        if cancel:
            for task in list(self._inflight_speculations.values()):
                task.cancel()
        await self.drain()

    def _observe_and_cross_predict(
        self,
        tool_name: str,
        args: Mapping[str, Any],
    ) -> None:
        self.predictor.observe(tool_name)
        if self.idempotency.is_safe_for_speculation(tool_name):
            self.predictor.observe_args(tool_name, args)

        guessed = self.predictor.guess_next_with_confidence(tool_name)
        if guessed is None:
            return
        next_tool, confidence = guessed
        if confidence < self.min_next_tool_confidence:
            return
        if not self.idempotency.is_safe_for_speculation(next_tool):
            return
        next_args = self.predictor.guess_args(next_tool)
        if next_args is None:
            return
        self._try_start_speculation(next_tool, next_args, "cross_turn")

    def _try_start_speculation(
        self,
        tool_name: str,
        args: Mapping[str, Any],
        source: Literal["tool_start", "in_stream", "cross_turn"],
        *,
        call_id: str | None = None,
    ) -> None:
        if self.executor is None:
            return
        if self.rollout_mode != "speculate":
            return
        if not self.idempotency.is_safe_for_speculation(tool_name):
            return
        if self._is_speculation_paused():
            return

        normalized_args = dict(args)
        key = cache_key(tool_name, normalized_args)
        if self.cache.peek_key(key) is not None:
            return
        if key in self._inflight_speculations:
            return
        if len(self._inflight_speculations) >= self.max_concurrent_speculations:
            self.stats.bump("speculations_throttled")
            self._trace(
                "speculation_throttled",
                source=source,
                tool_name=tool_name,
                key=key,
            )
            return

        if source == "tool_start":
            self.stats.bump("tool_start_speculations")
        elif source == "in_stream":
            self.stats.bump("speculations_launched")
        else:
            self.stats.bump("cross_turn_speculations")

        self._log(f"spec LAUNCH source={source} tool={tool_name}")
        self._trace("speculation_launched", source=source, tool_name=tool_name, key=key)
        task = asyncio.create_task(
            self._run_speculation(key, tool_name, normalized_args),
        )
        self._inflight_speculations[key] = task
        if call_id is not None:
            self._call_speculation_keys.setdefault(call_id, set()).add(key)

    async def _run_speculation(
        self,
        key: str,
        tool_name: str,
        args: Mapping[str, Any],
    ) -> None:
        started_at = time.monotonic()
        try:
            assert self.executor is not None
            result = self.executor(tool_name, args)
            if inspect.isawaitable(result):
                result = await result
            elapsed_ms = (time.monotonic() - started_at) * 1000
            self.cache.set(
                tool_name,
                args,
                result,
                observed_latency_ms=elapsed_ms,
            )
            self.stats.bump("speculations_resolved")
            self._log(f"spec RESOLVED tool={tool_name} took={elapsed_ms:.0f}ms")
            self._trace(
                "speculation_resolved",
                tool_name=tool_name,
                key=key,
                elapsed_ms=elapsed_ms,
            )
        except asyncio.CancelledError:
            self.stats.bump("speculations_cancelled")
            self._log(f"spec CANCELLED tool={tool_name}")
            self._trace("speculation_cancelled", tool_name=tool_name, key=key)
            raise
        except Exception as exc:  # pragma: no cover - defensive logging path.
            self.stats.bump("speculation_errors")
            self._log(f"spec FAILED tool={tool_name} err={exc!r}")
            self._trace(
                "speculation_error",
                tool_name=tool_name,
                key=key,
                error=repr(exc),
            )
        finally:
            self._inflight_speculations.pop(key, None)

    def _cancel_wrong_call_speculations(self, call_id: str, actual_key: str) -> None:
        keys = self._call_speculation_keys.pop(call_id, set())
        for key in keys:
            if key == actual_key:
                continue
            self.stats.bump("wasted_speculations")
            self._trace("speculation_wasted", key=key, actual_key=actual_key)
            task = self._inflight_speculations.get(key)
            if task is not None and not task.done():
                task.cancel()

    def _is_speculation_paused(self) -> bool:
        return time.monotonic() < self._speculation_paused_until

    def _can_provide_cached_result(self) -> bool:
        return self.rollout_mode in {"memoize", "speculate"}

    def _maybe_pause_speculation(self) -> None:
        if self.adaptive_min_calls == 0:
            return
        if self._is_speculation_paused():
            return
        total = self.stats.stats.cache_hits + self.stats.stats.cache_misses
        if total < self.adaptive_min_calls:
            return
        if self.stats.hit_rate() >= self.adaptive_min_hit_rate:
            return
        paused_until = time.monotonic() + self.adaptive_cooldown_seconds
        if paused_until > self._speculation_paused_until:
            self._speculation_paused_until = paused_until
            self.stats.bump("adaptive_pauses")
            self._log(
                "spec PAUSED "
                f"hit_rate={self.stats.hit_rate():.2f} "
                f"cooldown={self.adaptive_cooldown_seconds:.0f}s"
            )
            self._trace(
                "speculation_paused",
                hit_rate=self.stats.hit_rate(),
                cooldown_seconds=self.adaptive_cooldown_seconds,
            )

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[precog] {message}")

    def _trace(self, event_type: str, **fields: Any) -> None:
        if self.trace_sink is not None:
            self.trace_sink.emit(event_type, runtime_id=self.id, **fields)


def _event_value(
    event: ModelEvent | Mapping[str, Any],
    snake_name: str,
    camel_name: str | None = None,
    default: Any = None,
) -> Any:
    if isinstance(event, ModelEvent):
        return getattr(event, snake_name, default)
    if snake_name in event:
        return event[snake_name]
    if camel_name and camel_name in event:
        return event[camel_name]
    return default
