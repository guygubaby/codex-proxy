from __future__ import annotations

import os
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask


DEFAULT_UPSTREAM_BASE_URL = ""
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MODEL_CREATED_AT = "2026-01-01T00:00:00Z"
DEFAULT_CONTEXT_WINDOW_SIZE = 200000

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "trailers",
    "transfer-encoding",
    "upgrade",
}

REQUEST_HEADERS_TO_DROP = HOP_BY_HOP_HEADERS | {
    "host",
    "content-length",
}

RESPONSE_HEADERS_TO_DROP = HOP_BY_HOP_HEADERS | {
    "content-length",
}


def _load_dotenv(path: str = ".env") -> None:
    """Small .env loader to avoid requiring python-dotenv for simple deploys."""
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as dotenv_file:
        for raw_line in dotenv_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


_load_dotenv()

logging.basicConfig(
    level=os.getenv("CODEX_PROXY_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("codex_proxy")


class Settings:
    def __init__(self) -> None:
        self.upstream_base_url = os.getenv(
            "CODEX_PROXY_UPSTREAM_BASE_URL",
            DEFAULT_UPSTREAM_BASE_URL,
        ).rstrip("/")
        self.anthropic_version = DEFAULT_ANTHROPIC_VERSION
        self.connect_timeout = _env_float("CODEX_PROXY_CONNECT_TIMEOUT", 30.0)
        self.write_timeout = _env_float("CODEX_PROXY_WRITE_TIMEOUT", 60.0)
        self.pool_timeout = _env_float("CODEX_PROXY_POOL_TIMEOUT", 30.0)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    timeout = httpx.Timeout(
        connect=settings.connect_timeout,
        read=None,
        write=settings.write_timeout,
        pool=settings.pool_timeout,
    )
    app.state.http_client = httpx.AsyncClient(
        follow_redirects=False,
        timeout=timeout,
    )
    try:
        yield
    finally:
        await app.state.http_client.aclose()


app = FastAPI(
    title="Codex API Proxy",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)


def _extract_bearer(headers: dict[str, str]) -> str:
    authorization = headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return ""


def _extract_request_api_key(request: Request) -> str:
    headers = {key.lower(): value for key, value in request.headers.items()}
    return _extract_bearer(headers) or headers.get("x-api-key", "")


def _normalize_proxy_path(full_path: str) -> str:
    return full_path.strip("/")


def _is_anthropic_compatible_path(path: str) -> bool:
    normalized = path.strip("/")
    return normalized.startswith("v1/messages") or normalized.startswith("v1/complete")


def _is_anthropic_request(request: Request) -> bool:
    return "anthropic-version" in {key.lower() for key in request.headers}


def _resolve_upstream_model(model: str) -> str:
    if model.startswith("anthropic/"):
        return model.split("/", 1)[1]
    return model


def _is_codex_upstream() -> bool:
    return "/backend-api/codex" in settings.upstream_base_url.lower()


def _codex_base_path_includes_responses() -> bool:
    return settings.upstream_base_url.rstrip("/").lower().endswith("/backend-api/codex/responses")


def _responses_upstream_path() -> str:
    if not _is_codex_upstream():
        return "v1/responses"
    return "" if _codex_base_path_includes_responses() else "responses"


def _proxy_upstream_path(path: str) -> str:
    normalized = _normalize_proxy_path(path)
    if not _is_codex_upstream():
        return normalized

    if normalized == "v1/responses":
        return _responses_upstream_path()

    if normalized.startswith("v1/responses/"):
        suffix = normalized[len("v1/responses/") :]
        return suffix if _codex_base_path_includes_responses() else f"responses/{suffix}"

    return normalized


def _build_upstream_url(path: str, query: str) -> httpx.URL:
    base = settings.upstream_base_url
    if not base:
        raise HTTPException(
            status_code=500,
            detail="CODEX_PROXY_UPSTREAM_BASE_URL is not configured",
        )
    normalized_path = _normalize_proxy_path(path)
    url = httpx.URL(f"{base}/{normalized_path}" if normalized_path else base)
    if query:
        url = url.copy_with(query=query.encode("utf-8"))
    return url


def _build_upstream_headers(request: Request, path: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key, value in request.headers.items():
        lower_key = key.lower()
        if lower_key in REQUEST_HEADERS_TO_DROP:
            continue
        headers[key] = value

    request_api_key = _extract_request_api_key(request)
    if request_api_key:
        headers["authorization"] = f"Bearer {request_api_key}"
        headers["x-api-key"] = request_api_key

    if _is_anthropic_compatible_path(path):
        headers.setdefault("anthropic-version", settings.anthropic_version)

    headers.setdefault("user-agent", "codex-proxy/0.1.0")
    return headers


def _apply_codex_responses_headers(headers: dict[str, str], payload: dict[str, Any] | None = None) -> None:
    if not _is_codex_upstream():
        return

    header_keys = {key.lower() for key in headers}
    if "openai-beta" not in header_keys:
        headers["openai-beta"] = "responses=experimental"
    if "originator" not in header_keys:
        headers["originator"] = "codex_cli_rs"
    headers["content-type"] = "application/json"

    header_keys = {key.lower() for key in headers}
    if payload and payload.get("stream") is True:
        headers["accept"] = "text/event-stream"
    elif payload is not None:
        headers["accept"] = "application/json"
    elif "accept" not in header_keys:
        headers["accept"] = "application/json"


def _is_responses_proxy_path(path: str) -> bool:
    normalized = _normalize_proxy_path(path)
    return (
        normalized == "v1/responses"
        or normalized.startswith("v1/responses/")
        or normalized == "responses"
        or normalized.startswith("responses/")
    )


def _filter_response_headers(headers: httpx.Headers) -> dict[str, str]:
    filtered: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in RESPONSE_HEADERS_TO_DROP:
            continue
        filtered[key] = value
    return filtered


def _truncate_log_value(value: str, limit: int = 1200) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "...<truncated>"


def _request_auth_source(request: Request) -> str:
    headers = {key.lower(): value for key, value in request.headers.items()}
    if _extract_bearer(headers):
        return "authorization"
    if headers.get("x-api-key"):
        return "x-api-key"
    return "missing"


def _log_request(request: Request, route: str, **fields: Any) -> None:
    logger.info(
        "request route=%s method=%s auth=%s %s",
        route,
        request.method,
        _request_auth_source(request),
        " ".join(f"{key}={value}" for key, value in fields.items()),
    )


def _log_upstream_response(route: str, upstream_response: httpx.Response, elapsed_ms: int) -> None:
    logger.info(
        "upstream route=%s status=%s elapsed_ms=%s request_id=%s content_type=%s",
        route,
        upstream_response.status_code,
        elapsed_ms,
        upstream_response.headers.get("x-request-id", ""),
        upstream_response.headers.get("content-type", ""),
    )


async def _upstream_error_response(
    route: str,
    upstream_response: httpx.Response,
    elapsed_ms: int,
) -> Response:
    body = await upstream_response.aread()
    logger.warning(
        "upstream_error route=%s status=%s elapsed_ms=%s request_id=%s body=%s",
        route,
        upstream_response.status_code,
        elapsed_ms,
        upstream_response.headers.get("x-request-id", ""),
        _truncate_log_value(body.decode("utf-8", errors="replace")),
    )
    await upstream_response.aclose()
    return Response(
        content=body,
        status_code=upstream_response.status_code,
        headers=_filter_response_headers(upstream_response.headers),
        media_type=upstream_response.headers.get("content-type"),
    )


def _model_context_source(raw_model: Any) -> str:
    if isinstance(raw_model, dict):
        for key in (
            "context_window_size",
            "context_window",
            "context_length",
            "max_context_window",
            "max_context_tokens",
            "max_input_tokens",
        ):
            value = _safe_int(raw_model.get(key), 0)
            if value > 0:
                return key
    return "default"


def _anthropic_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return "" if content is None else str(content)

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue

        if not isinstance(block, dict):
            continue

        block_type = block.get("type")
        if block_type == "text":
            parts.append(str(block.get("text", "")))
        elif block_type == "tool_use":
            tool_name = block.get("name", "")
            tool_input = json.dumps(block.get("input", {}), ensure_ascii=False)
            parts.append(f"Tool use {tool_name}: {tool_input}")
        elif block_type == "tool_result":
            tool_content = block.get("content", "")
            parts.append(f"Tool result for {block.get('tool_use_id', '')}: {_anthropic_content_to_text(tool_content)}")
        elif block_type == "image":
            parts.append("[image]")

    return "\n".join(part for part in parts if part)


def _anthropic_messages_to_responses_payload(
    payload: dict[str, Any],
    *,
    stream: bool,
    codex_compat: bool,
) -> dict[str, Any]:
    input_items: list[dict[str, Any]] = []
    instructions: list[str] = []

    system_text = _anthropic_content_to_text(payload.get("system", ""))
    if system_text:
        instructions.append(system_text)

    for message in payload.get("messages", []):
        if not isinstance(message, dict):
            continue

        role = message.get("role", "user")
        if role not in {"user", "assistant", "system", "developer"}:
            role = "user"

        if role in {"system", "developer"}:
            text = _anthropic_content_to_text(message.get("content", ""))
            if text:
                instructions.append(text)
            continue

        if role == "assistant":
            _append_responses_assistant_input(input_items, message.get("content", ""))
        else:
            _append_responses_user_input(input_items, message.get("content", ""))

    responses_payload: dict[str, Any] = {
        "model": _resolve_upstream_model(str(payload.get("model") or "gpt-5.3-codex")),
        "input": input_items,
        "stream": True if codex_compat else stream,
    }

    if instructions:
        responses_payload["instructions"] = "\n\n".join(instructions)
    elif codex_compat:
        responses_payload["instructions"] = ""

    max_tokens = payload.get("max_tokens")
    if isinstance(max_tokens, int) and max_tokens > 0:
        responses_payload["max_output_tokens"] = max_tokens

    temperature = payload.get("temperature")
    if isinstance(temperature, (int, float)):
        responses_payload["temperature"] = temperature

    top_p = payload.get("top_p")
    if isinstance(top_p, (int, float)):
        responses_payload["top_p"] = top_p

    tools = payload.get("tools")
    if isinstance(tools, list) and tools:
        responses_payload["tools"] = [
            {
                "type": "function",
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            }
            for tool in tools
            if isinstance(tool, dict) and tool.get("name")
        ]

    tool_choice = _responses_tool_choice(payload.get("tool_choice"))
    if tool_choice:
        responses_payload["tool_choice"] = tool_choice

    if codex_compat:
        responses_payload["store"] = False
        responses_payload.pop("max_output_tokens", None)
        responses_payload.pop("temperature", None)

    return responses_payload


def _append_responses_user_input(input_items: list[dict[str, Any]], content: Any) -> None:
    if not isinstance(content, list):
        text = _anthropic_content_to_text(content)
        if text:
            input_items.append({"role": "user", "content": [{"type": "input_text", "text": text}]})
        return

    parts: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue

        if block.get("type") == "tool_result":
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": str(block.get("tool_use_id", "")),
                    "output": _anthropic_content_to_text(block.get("content", "")),
                }
            )
            continue

        converted = _anthropic_block_to_responses_part(block, "user")
        if converted:
            parts.append(converted)

    if parts:
        input_items.append({"role": "user", "content": parts})


def _append_responses_assistant_input(input_items: list[dict[str, Any]], content: Any) -> None:
    if not isinstance(content, list):
        text = _anthropic_content_to_text(content)
        if text:
            input_items.append({"role": "assistant", "content": [{"type": "output_text", "text": text}]})
        return

    parts: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue

        if block.get("type") == "tool_use":
            tool_input = block.get("input", {})
            if not isinstance(tool_input, dict):
                tool_input = {}
            input_items.append(
                {
                    "type": "function_call",
                    "call_id": str(block.get("id", "")),
                    "name": str(block.get("name", "")),
                    "arguments": json.dumps(tool_input, ensure_ascii=False, separators=(",", ":")),
                }
            )
            continue

        converted = _anthropic_block_to_responses_part(block, "assistant")
        if converted:
            parts.append(converted)

    if parts:
        input_items.append({"role": "assistant", "content": parts})


def _anthropic_block_to_responses_part(block: dict[str, Any], role: str) -> dict[str, Any] | None:
    block_type = block.get("type")
    if block_type == "text":
        text = str(block.get("text", ""))
        if text:
            return {"type": "output_text" if role == "assistant" else "input_text", "text": text}
        return None

    if block_type == "thinking":
        text = str(block.get("thinking", ""))
        return {"type": "output_text", "text": text} if text else None

    if role == "user" and block_type == "image":
        source = block.get("source")
        if not isinstance(source, dict):
            return None

        if source.get("type") == "base64":
            media_type = str(source.get("media_type") or "image/jpeg")
            data = str(source.get("data") or "")
            if data:
                return {"type": "input_image", "image_url": f"data:{media_type};base64,{data}"}

        if source.get("type") == "url":
            url = str(source.get("url") or "")
            if url:
                return {"type": "input_image", "image_url": url}

    return None


def _responses_tool_choice(value: Any) -> Any:
    if not value:
        return None
    if value == "auto":
        return "auto"
    if value == "any":
        return "required"
    if value == "none":
        return "none"
    if isinstance(value, dict) and value.get("type") == "tool" and value.get("name"):
        return {"type": "function", "name": str(value["name"])}
    return None


def _sse_event(event: str | None, data: Any) -> bytes:
    lines: list[str] = []
    if event:
        lines.append(f"event: {event}")
    if isinstance(data, str):
        payload = data
    else:
        payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    for line in payload.splitlines() or [""]:
        lines.append(f"data: {line}")
    lines.append("")
    return ("\n".join(lines) + "\n").encode("utf-8")


async def _iter_sse_events(response: httpx.Response) -> AsyncIterator[tuple[str | None, str]]:
    event: str | None = None
    data_lines: list[str] = []

    async for raw_line in response.aiter_lines():
        line = raw_line.rstrip("\r")
        if not line:
            if data_lines:
                yield event, "\n".join(data_lines)
            event = None
            data_lines = []
            continue

        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    if data_lines:
        yield event, "\n".join(data_lines)


def _text_from_string_fields(data: Any, keys: tuple[str, ...]) -> str:
    if not isinstance(data, dict):
        return ""

    for key in keys:
        value = data.get(key)
        if isinstance(value, str):
            return value

    return ""


def _extract_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        text = _text_from_string_fields(content, ("text", "output_text", "content"))
        if text:
            return text
        return _extract_content_text(content.get("content"))

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            text = _extract_content_text(block)
            if text:
                parts.append(text)
        return "".join(parts)

    return ""


def _extract_choice_text(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list):
        return ""

    parts: list[str] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue

        delta_text = _extract_content_text(choice.get("delta"))
        if delta_text:
            parts.append(delta_text)
            continue

        text = _text_from_string_fields(choice, ("text", "content"))
        if text:
            parts.append(text)
            continue

        message_text = _extract_content_text(choice.get("message"))
        if message_text:
            parts.append(message_text)

    return "".join(parts)


def _is_openai_choice_delta(data: dict[str, Any]) -> bool:
    choices = data.get("choices")
    if not isinstance(choices, list):
        return False
    return any(isinstance(choice, dict) and isinstance(choice.get("delta"), dict) for choice in choices)


def _extract_response_item_text(item: Any) -> str:
    if not isinstance(item, dict):
        return ""

    text = _text_from_string_fields(item, ("text", "output_text"))
    if text:
        return text

    return _extract_content_text(item.get("content"))


def _extract_response_text(data: dict[str, Any]) -> str:
    delta = data.get("delta")
    if isinstance(delta, str):
        return delta
    if isinstance(delta, dict):
        text = _extract_content_text(delta)
        if text:
            return text

    choice_text = _extract_choice_text(data)
    if choice_text:
        return choice_text

    text = _text_from_string_fields(data, ("text", "output_text", "content"))
    if text:
        return text

    part_text = _extract_content_text(data.get("part"))
    if part_text:
        return part_text

    item_text = _extract_response_item_text(data.get("item"))
    if item_text:
        return item_text

    response = data.get("response")
    if isinstance(response, dict):
        output_text = response.get("output_text")
        if isinstance(output_text, str):
            return output_text

        output = response.get("output")
        if isinstance(output, list):
            parts: list[str] = []
            for item in output:
                text = _extract_response_item_text(item)
                if text:
                    parts.append(text)
            return "".join(parts)

    return ""


def _safe_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default


def _anthropic_usage_from_response_usage(usage: Any) -> dict[str, int]:
    if not isinstance(usage, dict):
        usage = {}

    total_input_tokens = _safe_int(usage.get("input_tokens", usage.get("prompt_tokens", 0)))
    output_tokens = _safe_int(usage.get("output_tokens", usage.get("completion_tokens", 0)))
    input_details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    cached_tokens = _safe_int(input_details.get("cached_tokens", 0)) if isinstance(input_details, dict) else 0

    return {
        "input_tokens": max(total_input_tokens - cached_tokens, 0),
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": cached_tokens,
        "output_tokens": output_tokens,
    }


def _extract_response_usage(data: dict[str, Any]) -> dict[str, int]:
    response = data.get("response")
    if isinstance(response, dict):
        return _anthropic_usage_from_response_usage(response.get("usage", {}))
    return _anthropic_usage_from_response_usage(data.get("usage", {}))


def _extract_response_failure(data: dict[str, Any]) -> dict[str, str]:
    response = data.get("response")
    response_status = str(response.get("status", "")) if isinstance(response, dict) else ""

    error = data.get("error")
    if not error and isinstance(response, dict):
        error = response.get("error") or response.get("incomplete_details")

    if isinstance(error, dict):
        message = str(error.get("message") or error.get("reason") or "Upstream response failed")
        return {
            "type": str(error.get("type") or data.get("type") or "api_error"),
            "code": str(error.get("code") or ""),
            "status": response_status,
            "message": _truncate_log_value(message, 1000),
        }

    if isinstance(error, str):
        return {
            "type": str(data.get("type") or "api_error"),
            "code": "",
            "status": response_status,
            "message": _truncate_log_value(error, 1000),
        }

    return {
        "type": str(data.get("type") or "api_error"),
        "code": "",
        "status": response_status,
        "message": "Upstream response failed",
    }


def _extract_tool_call_from_item(item: Any) -> dict[str, str] | None:
    if not isinstance(item, dict):
        return None

    item_type = item.get("type")
    if item_type not in {"function_call", "tool_call"}:
        return None

    name = item.get("name") or item.get("function", {}).get("name")
    if not name:
        return None

    arguments = item.get("arguments")
    if arguments is None:
        arguments = item.get("function", {}).get("arguments", "")
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    else:
        arguments = str(arguments or "")

    return {
        "item_id": str(item.get("id") or item.get("call_id") or name),
        "tool_use_id": str(item.get("call_id") or item.get("id") or f"toolu_{int(time.time() * 1000)}"),
        "name": str(name),
        "arguments": arguments,
    }


def _extract_completed_tool_calls(data: dict[str, Any]) -> list[dict[str, str]]:
    response = data.get("response")
    if not isinstance(response, dict):
        return []

    output = response.get("output", [])
    if not isinstance(output, list):
        return []

    tool_calls: list[dict[str, str]] = []
    for item in output:
        tool_call = _extract_tool_call_from_item(item)
        if tool_call:
            tool_calls.append(tool_call)
    return tool_calls


async def _anthropic_stream_from_responses(
    upstream_response: httpx.Response,
    model: str,
) -> AsyncIterator[bytes]:
    if not _is_event_stream(upstream_response.headers):
        body_text = (await upstream_response.aread()).decode("utf-8", errors="replace")
        try:
            payload = json.loads(body_text)
        except json.JSONDecodeError:
            payload = {"output_text": body_text, "usage": {}}

        async for chunk in _anthropic_stream_from_message_body(_response_payload_to_anthropic_message_body(payload, model)):
            yield chunk
        return

    message_id = f"msg_{int(time.time() * 1000)}"
    output_text = ""
    text_index: int | None = None
    next_content_index = 0
    tool_blocks: dict[str, dict[str, Any]] = {}
    tool_used = False
    final_usage = _anthropic_usage_from_response_usage({})
    emitted_text_chars = 0
    emitted_text_deltas = 0
    emitted_tool_blocks = 0

    async def emit_text_delta(text_delta: str) -> AsyncIterator[bytes]:
        nonlocal emitted_text_chars, emitted_text_deltas, output_text, text_index, next_content_index

        if not text_delta:
            return

        if text_index is None:
            text_index = next_content_index
            next_content_index += 1
            yield _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": text_index,
                    "content_block": {"type": "text", "text": ""},
                },
            )

        output_text += text_delta
        emitted_text_chars += len(text_delta)
        emitted_text_deltas += 1
        yield _sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": text_index,
                "delta": {"type": "text_delta", "text": text_delta},
            },
        )

    async def emit_text_snapshot(text_snapshot: str, *, allow_disjoint: bool = False) -> AsyncIterator[bytes]:
        if not text_snapshot:
            return

        if text_snapshot.startswith(output_text):
            async for chunk in emit_text_delta(text_snapshot[len(output_text) :]):
                yield chunk
            return

        if not output_text or (allow_disjoint and text_snapshot not in output_text):
            async for chunk in emit_text_delta(text_snapshot):
                yield chunk

    async def start_tool_block(tool_call: dict[str, str]) -> AsyncIterator[bytes]:
        nonlocal emitted_tool_blocks, next_content_index, tool_used

        item_id = tool_call["item_id"]
        if item_id not in tool_blocks:
            tool_blocks[item_id] = {
                "index": next_content_index,
                "tool_use_id": tool_call["tool_use_id"],
                "name": tool_call["name"],
                "arguments": "",
                "started": False,
                "stopped": False,
            }
            next_content_index += 1

        block = tool_blocks[item_id]
        if not block["started"]:
            block["started"] = True
            tool_used = True
            emitted_tool_blocks += 1
            yield _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": block["index"],
                    "content_block": {
                        "type": "tool_use",
                        "id": block["tool_use_id"],
                        "name": block["name"],
                        "input": {},
                    },
                },
            )

    async def emit_tool_arguments(tool_call: dict[str, str], arguments_delta: str) -> AsyncIterator[bytes]:
        item_id = tool_call["item_id"]
        async for chunk in start_tool_block(tool_call):
            yield chunk

        if not arguments_delta:
            return

        block = tool_blocks[item_id]
        block["arguments"] += arguments_delta
        yield _sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": block["index"],
                "delta": {"type": "input_json_delta", "partial_json": arguments_delta},
            },
        )

    async def stop_tool_block(tool_call: dict[str, str]) -> AsyncIterator[bytes]:
        item_id = tool_call["item_id"]
        async for chunk in start_tool_block(tool_call):
            yield chunk

        block = tool_blocks[item_id]
        if not block["arguments"] and tool_call.get("arguments"):
            async for chunk in emit_tool_arguments(tool_call, tool_call["arguments"]):
                yield chunk

        if not block["stopped"]:
            block["stopped"] = True
            yield _sse_event(
                "content_block_stop",
                {"type": "content_block_stop", "index": block["index"]},
            )

    yield _sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": final_usage,
            },
        },
    )

    seen_event_types: set[str] = set()

    try:
        async for event, raw_data in _iter_sse_events(upstream_response):
            if raw_data == "[DONE]":
                continue

            try:
                data = json.loads(raw_data)
            except json.JSONDecodeError:
                continue

            event_type = data.get("type") or data.get("object") or event
            seen_event_types.add(str(event_type or "(none)"))
            if event_type in {"response.output_text.delta", "response.text.delta", "response.content_part.delta"}:
                async for chunk in emit_text_delta(_extract_response_text(data)):
                    yield chunk

            elif _is_openai_choice_delta(data):
                async for chunk in emit_text_delta(_extract_response_text(data)):
                    yield chunk

            elif event_type in {"response.output_text.done", "response.text.done"}:
                async for chunk in emit_text_snapshot(_extract_response_text(data)):
                    yield chunk

            elif event_type in {"response.content_part.done", "response.content_part.added"}:
                async for chunk in emit_text_snapshot(_extract_response_text(data)):
                    yield chunk

            elif event_type == "response.output_item.added":
                tool_call = _extract_tool_call_from_item(data.get("item"))
                if tool_call:
                    async for chunk in start_tool_block(tool_call):
                        yield chunk

            elif event_type == "response.function_call_arguments.delta":
                item_id = str(data.get("item_id") or data.get("output_item_id") or data.get("call_id") or "")
                tool_call = tool_blocks.get(item_id)
                if tool_call:
                    normalized_tool_call = {
                        "item_id": item_id,
                        "tool_use_id": str(tool_call["tool_use_id"]),
                        "name": str(tool_call["name"]),
                        "arguments": "",
                    }
                    async for chunk in emit_tool_arguments(normalized_tool_call, str(data.get("delta", ""))):
                        yield chunk

            elif event_type == "response.function_call_arguments.done":
                item_id = str(data.get("item_id") or data.get("output_item_id") or data.get("call_id") or "")
                block = tool_blocks.get(item_id)
                if block:
                    tool_call = {
                        "item_id": item_id,
                        "tool_use_id": str(block["tool_use_id"]),
                        "name": str(block["name"]),
                        "arguments": str(data.get("arguments", "")),
                    }
                    async for chunk in stop_tool_block(tool_call):
                        yield chunk

            elif event_type == "response.output_item.done":
                tool_call = _extract_tool_call_from_item(data.get("item"))
                if tool_call:
                    async for chunk in stop_tool_block(tool_call):
                        yield chunk
                else:
                    async for chunk in emit_text_snapshot(_extract_response_text(data), allow_disjoint=True):
                        yield chunk

            elif event_type == "response.completed":
                async for chunk in emit_text_snapshot(_extract_response_text(data), allow_disjoint=True):
                    yield chunk

                for tool_call in _extract_completed_tool_calls(data):
                    async for chunk in stop_tool_block(tool_call):
                        yield chunk

                final_usage = _extract_response_usage(data)

            elif event_type in {"response.failed", "response.incomplete", "error"}:
                failure = _extract_response_failure(data)
                logger.warning(
                    "upstream_response_failed model=%s event_type=%s status=%s code=%s message=%s",
                    model,
                    event_type,
                    failure["status"],
                    failure["code"],
                    failure["message"],
                )
                yield _sse_event(
                    "error",
                    {
                        "type": "error",
                        "error": {"type": "api_error", "message": failure["message"]},
                    },
                )
                return

        if text_index is not None:
            yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": text_index})

        for block in list(tool_blocks.values()):
            if not block["stopped"]:
                block["stopped"] = True
                yield _sse_event(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": block["index"]},
                )

        if text_index is None and not tool_used:
            logger.warning(
                "upstream stream produced no Anthropic content event_types=%s",
                ",".join(sorted(seen_event_types)) or "-",
            )
            yield _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            )
            yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": 0})

        logger.info(
            "stream_convert model=%s text_chars=%s text_deltas=%s tool_blocks=%s event_types=%s",
            model,
            emitted_text_chars,
            emitted_text_deltas,
            emitted_tool_blocks,
            ",".join(sorted(seen_event_types)) or "-",
        )

        yield _sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": "tool_use" if tool_used else "end_turn",
                    "stop_sequence": None,
                },
                "usage": final_usage,
            },
        )
        yield _sse_event("message_stop", {"type": "message_stop"})
    except httpx.HTTPError as exc:
        yield _sse_event(
            "error",
            {
                "type": "error",
                "error": {"type": "api_error", "message": f"Upstream stream failed: {exc}"},
            },
        )


async def _anthropic_stream_from_message_body(message: dict[str, Any]) -> AsyncIterator[bytes]:
    yield _sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                **message,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
            },
        },
    )

    content = message.get("content", [])
    if not isinstance(content, list):
        content = []

    for index, block in enumerate(content):
        if not isinstance(block, dict):
            continue

        if block.get("type") == "tool_use":
            yield _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": block.get("id", ""),
                        "name": block.get("name", ""),
                        "input": {},
                    },
                },
            )
            yield _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(block.get("input", {}), ensure_ascii=False, separators=(",", ":")),
                    },
                },
            )
            yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": index})
            continue

        yield _sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "text", "text": ""},
            },
        )
        text = str(block.get("text", ""))
        if text:
            yield _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": text},
                },
            )
        yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": index})

    yield _sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": message.get("stop_reason") or "end_turn",
                "stop_sequence": None,
            },
            "usage": message.get("usage") or _anthropic_usage_from_response_usage({}),
        },
    )
    yield _sse_event("message_stop", {"type": "message_stop"})


async def _anthropic_message_from_responses(upstream_response: httpx.Response, model: str) -> Response:
    if not _is_event_stream(upstream_response.headers):
        body_text = (await upstream_response.aread()).decode("utf-8", errors="replace")
        try:
            payload = json.loads(body_text)
        except json.JSONDecodeError:
            return JSONResponse(
                {
                    "id": f"msg_{int(time.time() * 1000)}",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [{"type": "text", "text": body_text}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": _anthropic_usage_from_response_usage({}),
                }
            )

        return JSONResponse(_response_payload_to_anthropic_message_body(payload, model))

    output_text = ""
    usage = _anthropic_usage_from_response_usage({})
    tool_calls: list[dict[str, str]] = []
    seen_event_types: set[str] = set()
    upstream_failure: dict[str, str] | None = None

    def append_text_snapshot(text_snapshot: str, *, allow_disjoint: bool = False) -> None:
        nonlocal output_text

        if not text_snapshot:
            return

        if text_snapshot.startswith(output_text):
            output_text += text_snapshot[len(output_text) :]
        elif not output_text or (allow_disjoint and text_snapshot not in output_text):
            output_text += text_snapshot

    async for event, raw_data in _iter_sse_events(upstream_response):
        if raw_data == "[DONE]":
            continue
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError:
            continue

        event_type = data.get("type") or data.get("object") or event
        seen_event_types.add(str(event_type or "(none)"))
        if event_type in {"response.output_text.delta", "response.text.delta", "response.content_part.delta"}:
            output_text += _extract_response_text(data)
        elif _is_openai_choice_delta(data):
            output_text += _extract_response_text(data)
        elif event_type in {"response.output_text.done", "response.text.done"}:
            append_text_snapshot(_extract_response_text(data))
        elif event_type in {"response.content_part.done", "response.content_part.added"}:
            append_text_snapshot(_extract_response_text(data))
        elif event_type == "response.output_item.done":
            tool_call = _extract_tool_call_from_item(data.get("item"))
            if tool_call:
                tool_calls.append(tool_call)
            else:
                append_text_snapshot(_extract_response_text(data), allow_disjoint=True)
        elif event_type == "response.completed":
            append_text_snapshot(_extract_response_text(data), allow_disjoint=True)
            usage = _extract_response_usage(data)
            tool_calls.extend(_extract_completed_tool_calls(data))
        elif event_type in {"response.failed", "response.incomplete", "error"}:
            upstream_failure = _extract_response_failure(data)

    logger.info(
        "message_convert model=%s text_chars=%s tool_calls=%s event_types=%s",
        model,
        len(output_text),
        len(tool_calls),
        ",".join(sorted(seen_event_types)) or "-",
    )

    if upstream_failure:
        logger.warning(
            "upstream_response_failed model=%s event_types=%s status=%s code=%s message=%s",
            model,
            ",".join(sorted(seen_event_types)) or "-",
            upstream_failure["status"],
            upstream_failure["code"],
            upstream_failure["message"],
        )
        return JSONResponse(
            {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": upstream_failure["message"],
                },
            },
            status_code=502,
        )

    content: list[dict[str, Any]] = []
    if output_text:
        content.append({"type": "text", "text": output_text})

    seen_tool_ids: set[str] = set()
    for tool_call in tool_calls:
        tool_use_id = tool_call["tool_use_id"]
        if tool_use_id in seen_tool_ids:
            continue
        seen_tool_ids.add(tool_use_id)
        try:
            tool_input = json.loads(tool_call.get("arguments") or "{}")
        except json.JSONDecodeError:
            tool_input = {"arguments": tool_call.get("arguments", "")}
        content.append(
            {
                "type": "tool_use",
                "id": tool_use_id,
                "name": tool_call["name"],
                "input": tool_input,
            }
        )

    return JSONResponse(
        {
            "id": f"msg_{int(time.time() * 1000)}",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content or [{"type": "text", "text": ""}],
            "stop_reason": "tool_use" if tool_calls else "end_turn",
            "stop_sequence": None,
            "usage": usage,
        }
    )


def _is_event_stream(headers: httpx.Headers) -> bool:
    return "text/event-stream" in headers.get("content-type", "").lower()


def _response_payload_to_anthropic_message_body(payload: Any, model: str) -> dict[str, Any]:
    output_text = _extract_response_payload_text(payload)
    tool_calls = _extract_completed_tool_calls({"response": payload})
    usage = _anthropic_usage_from_response_usage(payload.get("usage", {}) if isinstance(payload, dict) else {})
    response_model = str(payload.get("model") or model) if isinstance(payload, dict) else model

    content: list[dict[str, Any]] = []
    if output_text:
        content.append({"type": "text", "text": output_text})

    seen_tool_ids: set[str] = set()
    for tool_call in tool_calls:
        tool_use_id = tool_call["tool_use_id"]
        if tool_use_id in seen_tool_ids:
            continue
        seen_tool_ids.add(tool_use_id)

        try:
            tool_input = json.loads(tool_call.get("arguments") or "{}")
        except json.JSONDecodeError:
            tool_input = {"arguments": tool_call.get("arguments", "")}

        content.append(
            {
                "type": "tool_use",
                "id": tool_use_id,
                "name": tool_call["name"],
                "input": tool_input,
            }
        )

    return {
        "id": str(payload.get("id") or f"msg_{int(time.time() * 1000)}") if isinstance(payload, dict) else f"msg_{int(time.time() * 1000)}",
        "type": "message",
        "role": "assistant",
        "model": response_model,
        "content": content or [{"type": "text", "text": ""}],
        "stop_reason": "tool_use" if tool_calls else "end_turn",
        "stop_sequence": None,
        "usage": usage,
    }


def _extract_response_payload_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""

    output_text = payload.get("output_text")
    if isinstance(output_text, str):
        return output_text

    output = payload.get("output")
    if not isinstance(output, list):
        return ""

    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue

        content = item.get("content")
        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
                continue
            output_text = block.get("output_text")
            if isinstance(output_text, str):
                parts.append(output_text)

    return "".join(parts)


def _model_id_from_raw(raw_model: Any) -> str:
    if isinstance(raw_model, str):
        return raw_model
    if not isinstance(raw_model, dict):
        return ""
    return str(raw_model.get("id") or raw_model.get("slug") or raw_model.get("name") or "")


def _display_name_from_raw(raw_model: Any, model_id: str) -> str:
    if isinstance(raw_model, dict):
        value = raw_model.get("display_name") or raw_model.get("name") or raw_model.get("id") or raw_model.get("slug")
        if value:
            return str(value)
    return model_id


def _created_at_from_raw(raw_model: Any) -> str:
    if isinstance(raw_model, dict):
        created_at = raw_model.get("created_at")
        if isinstance(created_at, str) and created_at:
            return created_at

        created = raw_model.get("created")
        if isinstance(created, (int, float)):
            return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(created))

    return DEFAULT_MODEL_CREATED_AT


def _context_window_from_raw(raw_model: Any) -> int:
    if isinstance(raw_model, dict):
        for key in (
            "context_window_size",
            "context_window",
            "context_length",
            "max_context_window",
            "max_context_tokens",
            "max_input_tokens",
        ):
            value = _safe_int(raw_model.get(key), 0)
            if value > 0:
                return value
    return DEFAULT_CONTEXT_WINDOW_SIZE


def _model_list_from_payload(payload: Any) -> list[Any]:
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            return payload["data"]
        if isinstance(payload.get("models"), list):
            return payload["models"]
    if isinstance(payload, list):
        return payload
    return []


def _to_anthropic_models_response(payload: Any) -> dict[str, Any]:
    data: list[dict[str, Any]] = []
    for raw_model in _model_list_from_payload(payload):
        upstream_model_id = _model_id_from_raw(raw_model)
        if not upstream_model_id:
            continue

        model_id = upstream_model_id
        context_window_size = _context_window_from_raw(raw_model)
        data.append(
            {
                "type": "model",
                "id": model_id,
                "display_name": _display_name_from_raw(raw_model, upstream_model_id),
                "created_at": _created_at_from_raw(raw_model),
                "max_input_tokens": context_window_size,
                "context_window_size": context_window_size,
            }
        )

    return {
        "data": data,
        "has_more": False,
        "first_id": data[0]["id"] if data else None,
        "last_id": data[-1]["id"] if data else None,
    }


def _to_openai_models_response(payload: Any) -> dict[str, Any]:
    data: list[dict[str, Any]] = []
    for raw_model in _model_list_from_payload(payload):
        model_id = _model_id_from_raw(raw_model)
        if not model_id:
            continue

        created_at = _created_at_from_raw(raw_model)
        created = int(time.time())
        try:
            created = int(time.mktime(time.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")))
        except ValueError:
            pass

        data.append(
            {
                "id": model_id,
                "object": "model",
                "created": created,
                "owned_by": "upstream",
                "display_name": _display_name_from_raw(raw_model, model_id),
            }
        )

    return {"object": "list", "data": data}


@app.post("/v1/messages")
async def anthropic_messages(request: Request) -> Response:
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    model = str(payload.get("model") or "gpt-5.3-codex")
    wants_stream = bool(payload.get("stream", False))
    _log_request(
        request,
        "v1/messages",
        model=model,
        upstream_model=_resolve_upstream_model(model),
        stream=wants_stream,
        messages=len(payload.get("messages", [])) if isinstance(payload.get("messages"), list) else 0,
        tools=len(payload.get("tools", [])) if isinstance(payload.get("tools"), list) else 0,
    )
    upstream_path = _responses_upstream_path()
    upstream_payload = _anthropic_messages_to_responses_payload(
        payload,
        stream=wants_stream,
        codex_compat=_is_codex_upstream(),
    )
    upstream_url = _build_upstream_url(upstream_path, "")
    upstream_headers = _build_upstream_headers(request, upstream_path)
    upstream_headers["content-type"] = "application/json"
    _apply_codex_responses_headers(upstream_headers, upstream_payload)

    client: httpx.AsyncClient = request.app.state.http_client
    upstream_request = client.build_request(
        "POST",
        upstream_url,
        headers=upstream_headers,
        content=json.dumps(upstream_payload, ensure_ascii=False).encode("utf-8"),
    )

    try:
        start = time.monotonic()
        upstream_response = await client.send(upstream_request, stream=True)
        elapsed_ms = int((time.monotonic() - start) * 1000)
        route = f"v1/messages->{upstream_path or '(base)'}"
        _log_upstream_response(route, upstream_response, elapsed_ms)
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Upstream request timed out") from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc}") from exc

    if upstream_response.status_code >= 400:
        return await _upstream_error_response(route, upstream_response, elapsed_ms)

    if wants_stream:
        return StreamingResponse(
            _anthropic_stream_from_responses(upstream_response, model),
            media_type="text/event-stream",
            background=BackgroundTask(upstream_response.aclose),
        )

    response = await _anthropic_message_from_responses(upstream_response, model)
    await upstream_response.aclose()
    return response


@app.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(request: Request) -> dict[str, int]:
    payload = await request.json()
    _log_request(
        request,
        "v1/messages/count_tokens",
        model=payload.get("model", ""),
        messages=len(payload.get("messages", [])) if isinstance(payload.get("messages"), list) else 0,
    )
    text = _anthropic_content_to_text(payload.get("system", ""))
    for message in payload.get("messages", []):
        if isinstance(message, dict):
            text += "\n" + _anthropic_content_to_text(message.get("content", ""))

    return {"input_tokens": max(1, len(text) // 4)}


@app.get("/v1/models")
async def list_models(request: Request) -> Response:
    _log_request(request, "v1/models", mode="anthropic" if _is_anthropic_request(request) else "openai")
    upstream_url = _build_upstream_url("v1/models", request.url.query)
    upstream_headers = _build_upstream_headers(request, "v1/models")

    client: httpx.AsyncClient = request.app.state.http_client
    try:
        start = time.monotonic()
        upstream_response = await client.get(upstream_url, headers=upstream_headers)
        elapsed_ms = int((time.monotonic() - start) * 1000)
        _log_upstream_response("v1/models", upstream_response, elapsed_ms)
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Upstream request timed out") from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc}") from exc

    if upstream_response.status_code >= 400:
        return await _upstream_error_response("v1/models", upstream_response, elapsed_ms)

    try:
        payload = upstream_response.json()
    except json.JSONDecodeError:
        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            headers=_filter_response_headers(upstream_response.headers),
            media_type=upstream_response.headers.get("content-type"),
        )

    raw_models = _model_list_from_payload(payload)
    context_sources: dict[str, int] = {}
    for raw_model in raw_models:
        context_source = _model_context_source(raw_model)
        context_sources[context_source] = context_sources.get(context_source, 0) + 1
    logger.info(
        "models_transform mode=%s count=%s context_sources=%s",
        "anthropic" if _is_anthropic_request(request) else "openai",
        len(raw_models),
        context_sources,
    )

    if _is_anthropic_request(request):
        return JSONResponse(_to_anthropic_models_response(payload))

    return JSONResponse(_to_openai_models_response(payload))


async def _proxy(request: Request, full_path: str) -> Response:
    path = _normalize_proxy_path(full_path)
    if not path:
        return JSONResponse({"ok": True})

    _log_request(request, path, query=bool(request.url.query))
    upstream_path = _proxy_upstream_path(path)
    upstream_url = _build_upstream_url(upstream_path, request.url.query)
    upstream_headers = _build_upstream_headers(request, upstream_path)
    if _is_responses_proxy_path(path) or _is_responses_proxy_path(upstream_path):
        _apply_codex_responses_headers(upstream_headers)
    body = await request.body()

    client: httpx.AsyncClient = request.app.state.http_client
    upstream_request = client.build_request(
        request.method,
        upstream_url,
        headers=upstream_headers,
        content=body,
    )

    try:
        start = time.monotonic()
        upstream_response = await client.send(upstream_request, stream=True)
        elapsed_ms = int((time.monotonic() - start) * 1000)
        _log_upstream_response(path, upstream_response, elapsed_ms)
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Upstream request timed out") from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc}") from exc

    if upstream_response.status_code >= 400:
        return await _upstream_error_response(path, upstream_response, elapsed_ms)

    return StreamingResponse(
        upstream_response.aiter_raw(),
        status_code=upstream_response.status_code,
        headers=_filter_response_headers(upstream_response.headers),
        background=BackgroundTask(upstream_response.aclose),
    )


@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "ok": True,
        "has_upstream_base_url": bool(settings.upstream_base_url),
        "auth_source": "request_headers",
    }


@app.get("/healthz")
async def healthz() -> dict[str, object]:
    return await health()


@app.api_route(
    "/{full_path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def proxy_all(request: Request, full_path: str) -> Response:
    return await _proxy(request, full_path)
