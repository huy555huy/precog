from __future__ import annotations

import asyncio
from dataclasses import dataclass
import inspect
from typing import Any, Callable, Mapping

from .idempotency import IdempotencyClass


ToolFunction = Callable[..., Any]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    func: ToolFunction
    idempotency_class: IdempotencyClass = IdempotencyClass.UNKNOWN
    timeout_seconds: float | None = None
    run_sync_in_thread: bool = True


class ToolRegistry:
    """Small production-oriented registry for local Python tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(
        self,
        func: ToolFunction | None = None,
        *,
        name: str | None = None,
        idempotency_class: IdempotencyClass | str = IdempotencyClass.UNKNOWN,
        timeout_seconds: float | None = None,
        run_sync_in_thread: bool = True,
    ) -> ToolFunction | Callable[[ToolFunction], ToolFunction]:
        def decorator(inner: ToolFunction) -> ToolFunction:
            tool_name = name or inner.__name__
            self._tools[tool_name] = ToolSpec(
                name=tool_name,
                func=inner,
                idempotency_class=IdempotencyClass(idempotency_class),
                timeout_seconds=timeout_seconds,
                run_sync_in_thread=run_sync_in_thread,
            )
            return inner

        if func is None:
            return decorator
        return decorator(func)

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def idempotency_classes(self) -> dict[str, IdempotencyClass]:
        return {name: spec.idempotency_class for name, spec in self._tools.items()}

    def read_only_tools(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, spec in self._tools.items()
            if spec.idempotency_class
            in {IdempotencyClass.PURE_READ, IdempotencyClass.NETWORK_READ}
        )

    def precog_kwargs(self) -> dict[str, Any]:
        return {
            "idempotency_classes": self.idempotency_classes(),
            "executor": self.execute,
        }

    async def execute(self, tool_name: str, args: Mapping[str, Any]) -> Any:
        spec = self.get(tool_name)

        async def run() -> Any:
            if inspect.iscoroutinefunction(spec.func):
                return await spec.func(**dict(args))
            if spec.run_sync_in_thread:
                return await asyncio.to_thread(spec.func, **dict(args))
            return spec.func(**dict(args))

        if spec.timeout_seconds is None:
            return await run()
        return await asyncio.wait_for(run(), timeout=spec.timeout_seconds)

