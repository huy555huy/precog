from __future__ import annotations

import asyncio
from dataclasses import dataclass
import inspect
import json
import os
import threading
from typing import Any, AsyncIterator, Awaitable, Callable, Iterator, Mapping
from urllib import error, request

from ..runtime import ModelEvent, PreCog


ToolRunner = Callable[[str, Mapping[str, Any]], Any | Awaitable[Any]]
StreamEventHandler = Callable[[dict[str, Any]], None | Awaitable[None]]


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
        self.base_url = (
            base_url
            or os.getenv("ANTHROPIC_BASE_URL")
            or "https://api.anthropic.com"
        ).rstrip("/")
        self.auth_token = (
            auth_token
            or os.getenv("ANTHROPIC_AUTH_TOKEN")
            or os.getenv("ANTHROPIC_API_KEY")
        )
        self.model = model or os.getenv("ANTHROPIC_MODEL") or "claude-sonnet-4-5"
        self.timeout_seconds = timeout_seconds
        self.user_agent = (
            user_agent or os.getenv("ANTHROPIC_USER_AGENT") or "precog/0.6.0"
        )
        if not self.auth_token:
            raise RuntimeError(
                "missing Anthropic credentials; set ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY"
            )

    def messages_create(self, **payload: Any) -> dict[str, Any]:
        payload.setdefault("model", self.model)
        req = self._request(payload)
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

    async def messages_create_streaming(
        self,
        *,
        on_event: StreamEventHandler | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        """Create a streaming message and return the accumulated final message.

        Events are read on a background thread so async speculative tools can run
        while the HTTP response is still streaming.
        """

        accumulator = AnthropicStreamAccumulator()
        async for event in self.stream_events(**payload):
            if _get(event, "type") == "error":
                raise RuntimeError(f"Anthropic stream error: {_get(event, 'error')}")
            accumulator.add(event)
            if on_event is not None:
                result = on_event(event)
                if inspect.isawaitable(result):
                    await result
        return accumulator.final_message()

    async def stream_events(self, **payload: Any) -> AsyncIterator[dict[str, Any]]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any] | BaseException | None] = asyncio.Queue()

        def worker() -> None:
            try:
                for event in self.iter_events(**payload):
                    loop.call_soon_threadsafe(queue.put_nowait, event)
            except BaseException as exc:  # pragma: no cover - thread boundary.
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        thread = threading.Thread(
            target=worker,
            name="precog-anthropic-stream",
            daemon=True,
        )
        thread.start()
        while True:
            item = await queue.get()
            if item is None:
                await asyncio.to_thread(thread.join)
                return
            if isinstance(item, BaseException):
                await asyncio.to_thread(thread.join)
                raise item
            yield item

    def iter_events(self, **payload: Any) -> Iterator[dict[str, Any]]:
        payload.setdefault("model", self.model)
        payload["stream"] = True
        req = self._request(payload)
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                yield from _iter_sse_json(response)
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            raise RuntimeError(
                f"Anthropic Messages stream failed: HTTP {exc.code} {exc.reason}: {detail}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f"Anthropic Messages stream failed: {exc.reason}") from exc

    def _request(self, payload: Mapping[str, Any]) -> request.Request:
        url = f"{self.base_url}/v1/messages"
        body = json.dumps(payload).encode("utf-8")
        return request.Request(
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


class AnthropicStreamAccumulator:
    """Accumulate Anthropic SSE events into a Messages-style response dict."""

    def __init__(self) -> None:
        self._message: dict[str, Any] | None = None
        self._input_json: dict[int, str] = {}

    def add(self, event: Mapping[str, Any]) -> None:
        event_type = _get(event, "type")

        if event_type == "message_start":
            message = _get(event, "message", default={})
            self._message = dict(message) if isinstance(message, Mapping) else {}
            self._message.setdefault("content", [])
            return

        if event_type == "content_block_start":
            index = _get(event, "index")
            block = _get(event, "content_block", default={})
            if index is None or not isinstance(block, Mapping):
                return
            content = self._content()
            while len(content) <= int(index):
                content.append({})
            content[int(index)] = dict(block)
            if _get(block, "type") == "tool_use":
                self._input_json[int(index)] = ""
            return

        if event_type == "content_block_delta":
            index = _get(event, "index")
            delta = _get(event, "delta", default={})
            if index is None or not isinstance(delta, Mapping):
                return
            content = self._content()
            while len(content) <= int(index):
                content.append({})
            block = content[int(index)]
            delta_type = _get(delta, "type")
            if delta_type == "text_delta":
                block["text"] = str(block.get("text", "")) + str(
                    _get(delta, "text", default="")
                )
            elif delta_type == "input_json_delta":
                self._input_json[int(index)] = self._input_json.get(int(index), "") + str(
                    _get(delta, "partial_json", default="")
                )
            return

        if event_type == "content_block_stop":
            index = _get(event, "index")
            if index is None or int(index) not in self._input_json:
                return
            raw_input = self._input_json.pop(int(index))
            content = self._content()
            if int(index) >= len(content):
                return
            try:
                parsed = json.loads(raw_input or "{}")
            except json.JSONDecodeError:
                parsed = {"INVALID_JSON": raw_input}
            if isinstance(parsed, Mapping):
                content[int(index)]["input"] = dict(parsed)
            return

        if event_type == "message_delta":
            delta = _get(event, "delta", default={})
            if isinstance(delta, Mapping):
                self._message_dict().update(
                    {key: value for key, value in delta.items() if value is not None}
                )
            usage = _get(event, "usage")
            if usage is not None:
                self._message_dict()["usage"] = usage

    def final_message(self) -> dict[str, Any]:
        message = self._message_dict()
        if message.get("stop_reason") is None:
            content = message.get("content", [])
            if isinstance(content, list) and any(
                _get(block, "type") == "tool_use" for block in content
            ):
                message["stop_reason"] = "tool_use"
        return message

    def _message_dict(self) -> dict[str, Any]:
        if self._message is None:
            self._message = {"type": "message", "role": "assistant", "content": []}
        return self._message

    def _content(self) -> list[dict[str, Any]]:
        content = self._message_dict().setdefault("content", [])
        if not isinstance(content, list):
            content = []
            self._message_dict()["content"] = content
        return content


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


def tool_result_block(
    tool_use_id: str,
    result: Any,
    *,
    is_error: bool = False,
) -> dict[str, Any]:
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


def _iter_sse_json(response: Any) -> Iterator[dict[str, Any]]:
    data_lines: list[str] = []

    def flush() -> Iterator[dict[str, Any]]:
        nonlocal data_lines
        if data_lines:
            yield json.loads("\n".join(data_lines))
        data_lines = []

    def accept(line: str) -> Iterator[dict[str, Any]]:
        if line.endswith("\r"):
            line = line[:-1]
        if not line:
            yield from flush()
            return
        if line.startswith(":"):
            return
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    buffer = ""
    for raw_chunk in response:
        buffer += raw_chunk.decode("utf-8")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            yield from accept(line)
    if buffer:
        yield from accept(buffer)
    yield from flush()


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
