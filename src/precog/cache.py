from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
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

    def _is_expired(self, entry: CacheEntry) -> bool:
        return time.monotonic() - entry.inserted_at > self.ttl_seconds

    def _prune_expired(self) -> None:
        expired = [key for key, entry in self._store.items() if self._is_expired(entry)]
        for key in expired:
            self._store.pop(key, None)

