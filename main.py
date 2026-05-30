from __future__ import annotations

import os
import json
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask


DEFAULT_UPSTREAM_BASE_URL = ""
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"

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


class Settings:
    def __init__(self) -> None:
        self.upstream_base_url = os.getenv(
            "CODEX_PROXY_UPSTREAM_BASE_URL",
            DEFAULT_UPSTREAM_BASE_URL,
        ).rstrip("/")
        self.upstream_api_key = os.getenv("CODEX_PROXY_UPSTREAM_API_KEY", "")
        self.local_api_key = os.getenv("CODEX_PROXY_LOCAL_API_KEY", "")
        self.allow_unauthenticated = _env_bool(
            "CODEX_PROXY_ALLOW_UNAUTHENTICATED",
            default=not bool(self.local_api_key),
        )
        self.anthropic_version = os.getenv(
            "CODEX_PROXY_ANTHROPIC_VERSION",
            DEFAULT_ANTHROPIC_VERSION,
        )
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


def _authenticate_client(request: Request) -> None:
    if settings.allow_unauthenticated:
        return

    headers = {key.lower(): value for key, value in request.headers.items()}
    presented_key = _extract_bearer(headers) or headers.get("x-api-key", "")

    if not settings.local_api_key or presented_key != settings.local_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _normalize_proxy_path(full_path: str) -> str:
    return full_path.strip("/")


def _is_anthropic_compatible_path(path: str) -> bool:
    normalized = path.strip("/")
    return normalized.startswith("v1/messages") or normalized.startswith("v1/complete")


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

    if settings.upstream_api_key:
        headers["authorization"] = f"Bearer {settings.upstream_api_key}"
        headers["x-api-key"] = settings.upstream_api_key

    if _is_anthropic_compatible_path(path):
        headers.setdefault("anthropic-version", settings.anthropic_version)

    headers.setdefault("user-agent", "codex-proxy/0.1.0")
    return headers


def _filter_response_headers(headers: httpx.Headers) -> dict[str, str]:
    filtered: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in RESPONSE_HEADERS_TO_DROP:
            continue
        filtered[key] = value
    return filtered


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
        elif block_type == "tool_result":
            tool_content = block.get("content", "")
            parts.append(f"Tool result for {block.get('tool_use_id', '')}: {_anthropic_content_to_text(tool_content)}")
        elif block_type == "image":
            parts.append("[image]")

    return "\n".join(part for part in parts if part)


def _anthropic_messages_to_responses_payload(payload: dict[str, Any]) -> dict[str, Any]:
    input_items: list[dict[str, Any]] = []

    system = payload.get("system")
    if system:
        input_items.append(
            {
                "role": "system",
                "content": _anthropic_content_to_text(system),
            }
        )

    for message in payload.get("messages", []):
        if not isinstance(message, dict):
            continue

        role = message.get("role", "user")
        if role not in {"user", "assistant", "system", "developer"}:
            role = "user"

        input_items.append(
            {
                "role": role,
                "content": _anthropic_content_to_text(message.get("content", "")),
            }
        )

    responses_payload: dict[str, Any] = {
        "model": payload.get("model") or "gpt-5.3-codex",
        "input": input_items,
        "stream": True,
    }

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

    return responses_payload


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


def _extract_response_text(data: dict[str, Any]) -> str:
    delta = data.get("delta")
    if isinstance(delta, str):
        return delta

    text = data.get("text")
    if isinstance(text, str):
        return text

    response = data.get("response")
    if isinstance(response, dict):
        output = response.get("output")
        if isinstance(output, list):
            parts: list[str] = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                for content in item.get("content", []):
                    if isinstance(content, dict):
                        value = content.get("text")
                        if isinstance(value, str):
                            parts.append(value)
            return "".join(parts)

    return ""


async def _anthropic_stream_from_responses(
    upstream_response: httpx.Response,
    model: str,
) -> AsyncIterator[bytes]:
    message_id = f"msg_{int(time.time() * 1000)}"
    output_text = ""
    content_started = False

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
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )

    async for event, raw_data in _iter_sse_events(upstream_response):
        if raw_data == "[DONE]":
            continue

        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError:
            continue

        event_type = data.get("type") or event
        if event_type == "response.output_text.delta":
            text_delta = _extract_response_text(data)
            if not text_delta:
                continue

            if not content_started:
                content_started = True
                yield _sse_event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                )

            output_text += text_delta
            yield _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": text_delta},
                },
            )

        elif event_type == "response.completed":
            if not content_started and output_text:
                content_started = True
                yield _sse_event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
                yield _sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": output_text},
                    },
                )

            response = data.get("response", {})
            usage = response.get("usage", {}) if isinstance(response, dict) else {}
            if content_started:
                yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": 0})
            yield _sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {
                        "output_tokens": usage.get("output_tokens", 0),
                    },
                },
            )

    if not content_started:
        yield _sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        )
        yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": 0})

    yield _sse_event("message_stop", {"type": "message_stop"})


async def _anthropic_message_from_responses(upstream_response: httpx.Response, model: str) -> Response:
    output_text = ""
    usage: dict[str, Any] = {}

    async for event, raw_data in _iter_sse_events(upstream_response):
        if raw_data == "[DONE]":
            continue
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError:
            continue

        event_type = data.get("type") or event
        if event_type == "response.output_text.delta":
            output_text += _extract_response_text(data)
        elif event_type == "response.completed":
            response = data.get("response", {})
            if isinstance(response, dict):
                usage = response.get("usage", {}) or usage

    return JSONResponse(
        {
            "id": f"msg_{int(time.time() * 1000)}",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": output_text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            },
        }
    )


@app.post("/v1/messages")
async def anthropic_messages(request: Request) -> Response:
    _authenticate_client(request)

    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    model = str(payload.get("model") or "gpt-5.3-codex")
    wants_stream = bool(payload.get("stream", False))
    upstream_payload = _anthropic_messages_to_responses_payload(payload)
    upstream_url = _build_upstream_url("v1/responses", "")
    upstream_headers = _build_upstream_headers(request, "v1/responses")
    upstream_headers["content-type"] = "application/json"

    client: httpx.AsyncClient = request.app.state.http_client
    upstream_request = client.build_request(
        "POST",
        upstream_url,
        headers=upstream_headers,
        content=json.dumps(upstream_payload, ensure_ascii=False).encode("utf-8"),
    )

    try:
        upstream_response = await client.send(upstream_request, stream=True)
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Upstream request timed out") from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc}") from exc

    if upstream_response.status_code >= 400:
        return StreamingResponse(
            upstream_response.aiter_raw(),
            status_code=upstream_response.status_code,
            headers=_filter_response_headers(upstream_response.headers),
            background=BackgroundTask(upstream_response.aclose),
        )

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
    text = _anthropic_content_to_text(payload.get("system", ""))
    for message in payload.get("messages", []):
        if isinstance(message, dict):
            text += "\n" + _anthropic_content_to_text(message.get("content", ""))

    return {"input_tokens": max(1, len(text) // 4)}


async def _proxy(request: Request, full_path: str) -> Response:
    _authenticate_client(request)

    path = _normalize_proxy_path(full_path)
    if not path:
        return JSONResponse(
            {
                "ok": True,
                "upstream": settings.upstream_base_url,
                "routes": ["/*", "/v1/*"],
            }
        )

    upstream_url = _build_upstream_url(path, request.url.query)
    upstream_headers = _build_upstream_headers(request, path)
    body = await request.body()

    client: httpx.AsyncClient = request.app.state.http_client
    upstream_request = client.build_request(
        request.method,
        upstream_url,
        headers=upstream_headers,
        content=body,
    )

    try:
        upstream_response = await client.send(upstream_request, stream=True)
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Upstream request timed out") from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc}") from exc

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
        "upstream": settings.upstream_base_url,
        "has_upstream_base_url": bool(settings.upstream_base_url),
        "has_upstream_api_key": bool(settings.upstream_api_key),
        "allow_unauthenticated": settings.allow_unauthenticated,
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
