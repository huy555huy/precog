from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import time
from typing import Any, Mapping


@dataclass(frozen=True)
class ArgsRecord:
    args: Mapping[str, Any]
    observed_at: float


class ToolCallPredictor:
    """Tiny bigram predictor over recently observed tool calls."""

    def __init__(
        self,
        history_window: int = 32,
        min_observations: int = 1,
        args_window: int = 8,
    ) -> None:
        if history_window < 1:
            raise ValueError("history_window must be >= 1")
        if min_observations < 1:
            raise ValueError("min_observations must be >= 1")
        if args_window < 1:
            raise ValueError("args_window must be >= 1")
        self.history_window = history_window
        self.min_observations = min_observations
        self.args_window = args_window
        self._bigrams: dict[str, dict[str, int]] = {}
        self._args_memory: dict[str, list[ArgsRecord]] = {}
        self._last_tool: str | None = None

    def observe(self, tool_name: str) -> None:
        previous = self._last_tool
        if previous:
            row = self._bigrams.setdefault(previous, {})
            row[tool_name] = row.get(tool_name, 0) + 1
            self._trim(row)
        self._last_tool = tool_name

    def observe_args(self, tool_name: str, args: Mapping[str, Any]) -> None:
        records = self._args_memory.setdefault(tool_name, [])
        records.append(ArgsRecord(args=deepcopy(dict(args)), observed_at=time.monotonic()))
        del records[: max(0, len(records) - self.args_window)]

    def guess_next(self, previous: str) -> str | None:
        guessed = self.guess_next_with_confidence(previous)
        return guessed[0] if guessed is not None else None

    def guess_next_with_confidence(self, previous: str) -> tuple[str, float] | None:
        row = self._bigrams.get(previous)
        if not row:
            return None
        total = sum(row.values())
        if total < self.min_observations:
            return None
        name, count = max(row.items(), key=lambda item: item[1])
        return name, count / total

    def guess_args(self, tool_name: str) -> Mapping[str, Any] | None:
        records = self._args_memory.get(tool_name)
        if not records:
            return None
        return deepcopy(dict(records[-1].args))

    def to_dict(self) -> dict[str, Any]:
        return {
            "history_window": self.history_window,
            "min_observations": self.min_observations,
            "args_window": self.args_window,
            "bigrams": deepcopy(self._bigrams),
            "args_memory": {
                tool_name: [
                    {"args": deepcopy(dict(record.args)), "observed_at": record.observed_at}
                    for record in records
                ]
                for tool_name, records in self._args_memory.items()
            },
            "last_tool": self._last_tool,
        }

    def load_dict(self, state: Mapping[str, Any]) -> None:
        self._bigrams = {
            str(previous): {str(name): int(count) for name, count in row.items()}
            for previous, row in dict(state.get("bigrams", {})).items()
            if isinstance(row, Mapping)
        }
        self._args_memory = {}
        for tool_name, records in dict(state.get("args_memory", {})).items():
            loaded: list[ArgsRecord] = []
            if not isinstance(records, list):
                continue
            for record in records[-self.args_window :]:
                if not isinstance(record, Mapping):
                    continue
                args = record.get("args")
                if not isinstance(args, Mapping):
                    continue
                observed_at = float(record.get("observed_at", time.monotonic()))
                loaded.append(ArgsRecord(args=deepcopy(dict(args)), observed_at=observed_at))
            if loaded:
                self._args_memory[str(tool_name)] = loaded
        last_tool = state.get("last_tool")
        self._last_tool = str(last_tool) if last_tool is not None else None

    def _trim(self, row: dict[str, int]) -> None:
        total = sum(row.values())
        if total <= self.history_window:
            return
        for name, count in sorted(row.items(), key=lambda item: item[1]):
            if total <= self.history_window:
                break
            row.pop(name, None)
            total -= count
