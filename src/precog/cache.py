from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import time
from typing import Any, Iterable, Mapping


def stable_stringify(value: Any) -> str:
    """Return stable JSON for a JSON-like value."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def cache_key(tool_name: str, args: Any) -> str:
    """Canonical key for a tool name and argument object."""

    return f"{tool_name}::{stable_stringify(args)}"


_TOKEN_SPLIT_RE = re.compile(r"[\s\-_/.,;:?!()[\]{}]+")


def tokenize(value: Any) -> set[str]:
    """Tokenize nested JSON-like arguments for fuzzy cache lookup."""

    tokens: set[str] = set()

    def visit(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, str):
            for token in _TOKEN_SPLIT_RE.split(item.lower()):
                if token:
                    tokens.add(token)
            return
        if isinstance(item, bool | int | float):
            tokens.add(str(item))
            return
        if isinstance(item, Mapping):
            for child in item.values():
                visit(child)
            return
        if isinstance(item, Iterable) and not isinstance(item, bytes | bytearray):
            for child in item:
                visit(child)

    visit(value)
    return tokens


def jaccard(a: set[str] | frozenset[str], b: set[str] | frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    intersection = len(a.intersection(b))
    union = len(a) + len(b) - intersection
    return 0.0 if union == 0 else intersection / union


@dataclass(frozen=True)
class CacheEntry:
    key: str
    tool_name: str
    args: Mapping[str, Any]
    arg_tokens: frozenset[str]
    result: Any
    inserted_at: float
    observed_latency_ms: float


class SpeculationCache:
    """Small TTL-bound LRU cache for speculative tool results."""

    def __init__(self, max_entries: int = 64, ttl_seconds: float = 60.0) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()

    def set(
        self,
        tool_name: str,
        args: Mapping[str, Any],
        result: Any,
        observed_latency_ms: float = 0.0,
    ) -> str:
        key = cache_key(tool_name, args)
        self._store.pop(key, None)
        self._store[key] = CacheEntry(
            key=key,
            tool_name=tool_name,
            args=dict(args),
            arg_tokens=frozenset(tokenize(args)),
            result=result,
            inserted_at=time.monotonic(),
            observed_latency_ms=observed_latency_ms,
        )
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)
        return key

    def peek(self, tool_name: str, args: Mapping[str, Any]) -> CacheEntry | None:
        return self.peek_key(cache_key(tool_name, args))

    def peek_key(self, key: str) -> CacheEntry | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        if self._is_expired(entry):
            self._store.pop(key, None)
            return None
        self._store.move_to_end(key)
        return entry

    def fuzzy_find(
        self,
        tool_name: str,
        args: Mapping[str, Any],
        threshold: float = 0.85,
    ) -> tuple[CacheEntry, float] | None:
        query_tokens = tokenize(args)
        best: tuple[CacheEntry, float] | None = None
        expired_keys: list[str] = []

        for key, entry in self._store.items():
            if self._is_expired(entry):
                expired_keys.append(key)
                continue
            if entry.tool_name != tool_name:
                continue
            similarity = jaccard(query_tokens, entry.arg_tokens)
            if similarity < threshold:
                continue
            if best is None or similarity > best[1]:
                best = (entry, similarity)

        for key in expired_keys:
            self._store.pop(key, None)

        if best is not None:
            self._store.move_to_end(best[0].key)
        return best

    def size(self) -> int:
        self._prune_expired()
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()

    def to_dict(self) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        monotonic_now = time.monotonic()
        saved_at = time.time()
        for entry in list(self._store.values()):
            if self._is_expired(entry):
                continue
            age_seconds = monotonic_now - entry.inserted_at
            remaining_ttl_seconds = max(0.0, self.ttl_seconds - age_seconds)
            try:
                json.dumps(entry.result)
            except TypeError:
                continue
            entries.append(
                {
                    "tool_name": entry.tool_name,
                    "args": dict(entry.args),
                    "result": entry.result,
                    "saved_at": saved_at,
                    "remaining_ttl_seconds": remaining_ttl_seconds,
                    "observed_latency_ms": entry.observed_latency_ms,
                }
            )
        return {
            "max_entries": self.max_entries,
            "ttl_seconds": self.ttl_seconds,
            "entries": entries,
        }

    def load_dict(self, state: Mapping[str, Any]) -> None:
        for item in list(state.get("entries", [])):
            if not isinstance(item, Mapping):
                continue
            tool_name = item.get("tool_name")
            args = item.get("args")
            if not isinstance(tool_name, str) or not isinstance(args, Mapping):
                continue
            saved_at = float(item.get("saved_at", time.time()))
            remaining_ttl_seconds = item.get("remaining_ttl_seconds")
            if remaining_ttl_seconds is None:
                age_seconds = float(item.get("age_seconds", 0.0))
            else:
                elapsed_since_save = max(0.0, time.time() - saved_at)
                remaining = float(remaining_ttl_seconds) - elapsed_since_save
                if remaining <= 0:
                    continue
                age_seconds = self.ttl_seconds - remaining
            if age_seconds > self.ttl_seconds:
                continue
            key = self.set(
                tool_name,
                dict(args),
                item.get("result"),
                observed_latency_ms=float(item.get("observed_latency_ms", 0.0)),
            )
            entry = self._store[key]
            self._store[key] = CacheEntry(
                key=entry.key,
                tool_name=entry.tool_name,
                args=entry.args,
                arg_tokens=entry.arg_tokens,
                result=entry.result,
                inserted_at=time.monotonic() - max(0.0, age_seconds),
                observed_latency_ms=entry.observed_latency_ms,
            )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    def load(self, path: str | Path) -> None:
        self.load_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def _is_expired(self, entry: CacheEntry) -> bool:
        return time.monotonic() - entry.inserted_at > self.ttl_seconds

    def _prune_expired(self) -> None:
        expired = [key for key, entry in self._store.items() if self._is_expired(entry)]
        for key in expired:
            self._store.pop(key, None)
