"""PreCog: speculative tool execution for Python agent runtimes."""

from .cache import CacheEntry, SpeculationCache, cache_key, jaccard, tokenize
from .idempotency import IdempotencyClass, IdempotencyRegistry
from .predictor import ToolCallPredictor
from .runtime import (
    ModelEvent,
    PreCog,
    PreCogConfig,
    PreCogDecision,
    SpeculativeExecutor,
    ToolCall,
)
from .stats import PreCogStats, StatsCollector
from .tools import ToolRegistry, ToolSpec
from .trace import JsonlTraceSink, TraceSink

__all__ = [
    "CacheEntry",
    "IdempotencyClass",
    "IdempotencyRegistry",
    "ModelEvent",
    "PreCog",
    "PreCogConfig",
    "PreCogDecision",
    "PreCogStats",
    "SpeculationCache",
    "SpeculativeExecutor",
    "StatsCollector",
    "ToolCall",
    "ToolCallPredictor",
    "ToolRegistry",
    "ToolSpec",
    "JsonlTraceSink",
    "TraceSink",
    "cache_key",
    "jaccard",
    "tokenize",
]
