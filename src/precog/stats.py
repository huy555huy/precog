from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


StatsKey = Literal[
    "observed_tool_starts",
    "speculations_launched",
    "tool_start_speculations",
    "cross_turn_speculations",
    "speculations_resolved",
    "speculations_cancelled",
    "speculations_throttled",
    "speculation_errors",
    "cache_stores",
    "strict_hits",
    "fuzzy_hits",
    "cache_hits",
    "cache_misses",
    "wasted_speculations",
    "adaptive_pauses",
    "latency_saved_ms",
]


@dataclass
class PreCogStats:
    observed_tool_starts: int = 0
    speculations_launched: int = 0
    tool_start_speculations: int = 0
    cross_turn_speculations: int = 0
    speculations_resolved: int = 0
    speculations_cancelled: int = 0
    speculations_throttled: int = 0
    speculation_errors: int = 0
    cache_stores: int = 0
    strict_hits: int = 0
    fuzzy_hits: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    wasted_speculations: int = 0
    adaptive_pauses: int = 0
    latency_saved_ms: float = 0.0


class StatsCollector:
    def __init__(self) -> None:
        self.stats = PreCogStats()

    def bump(self, key: StatsKey, by: int | float = 1) -> None:
        setattr(self.stats, key, getattr(self.stats, key) + by)

    def snapshot(self) -> PreCogStats:
        return PreCogStats(**asdict(self.stats))

    def hit_rate(self) -> float:
        total = self.stats.cache_hits + self.stats.cache_misses
        return 0.0 if total == 0 else self.stats.cache_hits / total
