from __future__ import annotations

from enum import Enum
from typing import Iterable, Mapping


class IdempotencyClass(str, Enum):
    PURE_READ = "pure_read"
    NETWORK_READ = "network_read"
    LOCAL_WRITE_IDEMPOTENT = "local_write_idempotent"
    REMOTE_WRITE = "remote_write"
    UNKNOWN = "unknown"


SAFE_FOR_SPECULATION = {
    IdempotencyClass.PURE_READ,
    IdempotencyClass.NETWORK_READ,
}


class IdempotencyRegistry:
    """Classify tools so speculative execution only runs safe tools."""

    def __init__(
        self,
        read_only_tools: Iterable[str] | None = None,
        classes: Mapping[str, IdempotencyClass | str] | None = None,
    ) -> None:
        self._classes: dict[str, IdempotencyClass] = {}
        for name in read_only_tools or ():
            self._classes[name] = IdempotencyClass.PURE_READ
        for name, value in (classes or {}).items():
            self._classes[name] = IdempotencyClass(value)

    def classify(self, tool_name: str) -> IdempotencyClass:
        return self._classes.get(tool_name, IdempotencyClass.UNKNOWN)

    def is_safe_for_speculation(self, tool_name: str) -> bool:
        return self.classify(tool_name) in SAFE_FOR_SPECULATION

    def safe_tool_names(self) -> tuple[str, ...]:
        return tuple(
            name for name, cls in self._classes.items() if cls in SAFE_FOR_SPECULATION
        )
