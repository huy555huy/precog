from __future__ import annotations

from typing import Any, Mapping

from ..runtime import ModelEvent


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

