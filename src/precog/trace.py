from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import time
from typing import Any, Protocol


class TraceSink(Protocol):
    def emit(self, event_type: str, **fields: Any) -> None:
        """Record a runtime event."""


class JsonlTraceSink:
    """Append-only JSONL trace writer for production debugging."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event_type: str, **fields: Any) -> None:
        event = {
            "ts": time.time(),
            "event": event_type,
            **{key: _jsonable(value) for key, value in fields.items()},
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value

