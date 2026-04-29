from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Awaitable, Callable, Mapping
from urllib import error, request

from ..runtime import ModelEvent, PreCog


ToolRunner = Callable[[str, Mapping[str, Any]], Any | Awaitable[Any]]


@dataclass(frozen=True)
class AnthropicToolUse:
    id: str
    name: str
    input: dict[str, Any]


class AnthropicMessagesAdapter:
    """Translate Anthropic Messages streaming events into PreCog model events."""

    def __init__(self) -> None:
        self._index_to_tool: dict[int, tuple[str, str]] = {}

    def events_from(self, event: Any) -> list[ModelEvent]:
        event_type = _get(event, "type")
        events: list[ModelEvent] = []

        if event_type == "content_block_start":
            index = _get(event, "index")
            block = _get(event, "content_block", default={})
            if _get(block, "type") != "tool_use":
                return events
            tool_id = _get(block, "id")
            tool_name = _get(block, "name")
            if index is None or not tool_id or not tool_name:
                return events
            self._index_to_tool[int(index)] = (str(tool_id), str(tool_name))
            events.append(
                ModelEvent(
                    type="tool_call_start",
                    call_id=str(tool_id),
                    tool_name=str(tool_name),
                )
            )
            return events

        if event_type == "content_block_delta":
            index = _get(event, "index")
            if index is None or int(index) not in self._index_to_tool:
                return events
            delta = _get(event, "delta", default={})
            if _get(delta, "type") != "input_json_delta":
                return events
            partial_json = _get(delta, "partial_json", default="")
            call_id, _ = self._index_to_tool[int(index)]
            if partial_json:
                events.append(
                    ModelEvent(
                        type="tool_call_args_delta",
                        call_id=call_id,
                        delta=str(partial_json),
                    )
                )
            return events

        if event_type == "content_block_stop":
            index = _get(event, "index")
            if index is None:
                return events
            stored = self._index_to_tool.pop(int(index), None)
            if stored is None:
                return events
            call_id, _ = stored
            events.append(ModelEvent(type="tool_call_end", call_id=call_id))
            return events

        return events


class AnthropicMessagesClient:
    """Tiny stdlib Anthropic-compatible Messages client for evals."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        auth_token: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 60.0,
        user_agent: str | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/")
        self.auth_token = (
            auth_token
            or os.getenv("ANTHROPIC_AUTH_TOKEN")
            or os.getenv("ANTHROPIC_API_KEY")
        )
        self.model = model or os.getenv("ANTHROPIC_MODEL") or "claude-sonnet-4-5"
        self.timeout_seconds = timeout_seconds
        self.user_agent = user_agent or os.getenv("ANTHROPIC_USER_AGENT") or "precog/0.6.0"
        if not self.auth_token:
            raise RuntimeError(
                "missing Anthropic credentials; set ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY"
            )

    def messages_create(self, **payload: Any) -> dict[str, Any]:
        payload.setdefault("model", self.model)
        url = f"{self.base_url}/v1/messages"
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url,
            data=body,
            headers={
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
                "user-agent": self.user_agent,
                "x-api-key": self.auth_token or "",
                "authorization": f"Bearer {self.auth_token}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            raise RuntimeError(
                f"Anthropic Messages request failed: HTTP {exc.code} {exc.reason}: {detail}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f"Anthropic Messages request failed: {exc.reason}") from exc


def extract_tool_uses(message: Any) -> list[AnthropicToolUse]:
    uses: list[AnthropicToolUse] = []
    for block in list(_get(message, "content", default=[]) or []):
        if _get(block, "type") != "tool_use":
            continue
        tool_id = _get(block, "id")
        name = _get(block, "name")
        tool_input = _get(block, "input", default={})
        if not tool_id or not name:
            continue
        if not isinstance(tool_input, Mapping):
            tool_input = {}
        uses.append(
            AnthropicToolUse(
                id=str(tool_id),
                name=str(name),
                input=dict(tool_input),
            )
        )
    return uses


async def execute_message_tool_calls(
    precog: PreCog,
    message: Any,
    runner: ToolRunner,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for tool_use in extract_tool_uses(message):
        result = await precog.execute(
            tool_use.name,
            tool_use.input,
            runner,
            call_id=tool_use.id,
        )
        blocks.append(tool_result_block(tool_use.id, result))
    return blocks


def tool_result_block(tool_use_id: str, result: Any, *, is_error: bool = False) -> dict[str, Any]:
    if isinstance(result, str):
        content = result
    else:
        content = json.dumps(result, ensure_ascii=False, default=repr)
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }
    if is_error:
        block["is_error"] = True
    return block


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
