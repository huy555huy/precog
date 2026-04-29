from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Awaitable, Callable, Mapping

from ..runtime import ModelEvent, PreCog


ToolRunner = Callable[[str, Mapping[str, Any]], Any | Awaitable[Any]]


@dataclass(frozen=True)
class OpenAIFunctionCall:
    name: str
    arguments: dict[str, Any]
    call_id: str
    item_id: str | None = None


class OpenAIResponsesAdapter:
    """Translate OpenAI Responses streaming events into PreCog model events."""

    def __init__(self) -> None:
        self._item_to_call: dict[str, str] = {}

    def events_from(self, event: Any) -> list[ModelEvent]:
        event_type = _get(event, "type")
        events: list[ModelEvent] = []

        if event_type == "response.output_item.added":
            item = _get(event, "item", default={})
            item_type = _get(item, "type")
            if item_type not in {"function_call", "function"}:
                return events
            item_id = _get(item, "id") or _get(event, "item_id")
            call_id = _get(item, "call_id") or item_id
            tool_name = _get(item, "name") or _get(item, "function", "name")
            if item_id:
                self._item_to_call[str(item_id)] = str(call_id)
            if call_id and tool_name:
                events.append(
                    ModelEvent(
                        type="tool_call_start",
                        call_id=str(call_id),
                        tool_name=str(tool_name),
                    )
                )
            return events

        if event_type in {
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
        }:
            call_id = self._call_id_for(event)
            if not call_id:
                return events
            delta = _get(event, "delta")
            if delta is None and event_type.endswith(".done"):
                delta = _get(event, "arguments", default="")
            if delta:
                events.append(
                    ModelEvent(
                        type="tool_call_args_delta",
                        call_id=call_id,
                        delta=str(delta),
                    )
                )
            if event_type.endswith(".done"):
                events.append(ModelEvent(type="tool_call_end", call_id=call_id))
            return events

        if event_type == "response.output_item.done":
            item = _get(event, "item", default={})
            item_type = _get(item, "type")
            if item_type not in {"function_call", "function"}:
                return events
            call_id = _get(item, "call_id") or _get(item, "id")
            arguments = _get(item, "arguments")
            if call_id and arguments:
                events.append(
                    ModelEvent(
                        type="tool_call_args_delta",
                        call_id=str(call_id),
                        delta=str(arguments),
                    )
                )
            if call_id:
                events.append(ModelEvent(type="tool_call_end", call_id=str(call_id)))
            return events

        return events

    def _call_id_for(self, event: Any) -> str | None:
        call_id = _get(event, "call_id")
        if call_id:
            return str(call_id)
        item_id = _get(event, "item_id")
        if item_id:
            return self._item_to_call.get(str(item_id), str(item_id))
        return None


def events_from_openai_response_event(event: Any) -> list[ModelEvent]:
    return OpenAIResponsesAdapter().events_from(event)


def extract_function_calls(response_or_items: Any) -> list[OpenAIFunctionCall]:
    """Extract function calls from an OpenAI Responses object or output list."""

    items = _get(response_or_items, "output")
    if items is None:
        items = response_or_items
    calls: list[OpenAIFunctionCall] = []
    for item in list(items or []):
        if _get(item, "type") != "function_call":
            continue
        name = _get(item, "name")
        call_id = _get(item, "call_id")
        raw_args = _get(item, "arguments", default="{}")
        if not name or not call_id:
            continue
        if isinstance(raw_args, str):
            args = json.loads(raw_args or "{}")
        elif isinstance(raw_args, Mapping):
            args = dict(raw_args)
        else:
            args = {}
        if not isinstance(args, Mapping):
            args = {}
        calls.append(
            OpenAIFunctionCall(
                name=str(name),
                arguments=dict(args),
                call_id=str(call_id),
                item_id=_maybe_str(_get(item, "id")),
            )
        )
    return calls


async def execute_response_tool_calls(
    precog: PreCog,
    response_or_items: Any,
    runner: ToolRunner,
) -> list[dict[str, Any]]:
    """Execute OpenAI Responses function calls and return output input items."""

    outputs: list[dict[str, Any]] = []
    for call in extract_function_calls(response_or_items):
        result = await precog.execute(
            call.name,
            call.arguments,
            runner,
            call_id=call.call_id,
        )
        outputs.append(function_call_output(call.call_id, result))
    return outputs


def function_call_output(call_id: str, output: Any) -> dict[str, Any]:
    if isinstance(output, str):
        encoded = output
    else:
        encoded = json.dumps(output, ensure_ascii=False, default=repr)
    return {"type": "function_call_output", "call_id": call_id, "output": encoded}


def _get(obj: Any, *path: str, default: Any = None) -> Any:
    current = obj
    for key in path:
        if isinstance(current, Mapping):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)
        if current is default:
            return default
    return current


def _maybe_str(value: Any) -> str | None:
    return str(value) if value is not None else None
