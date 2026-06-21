const DEFAULT_UPSTREAM_BASE_URL = "";
const DEFAULT_ANTHROPIC_VERSION = "2023-06-01";
const DEFAULT_MODEL_CREATED_AT = "2026-01-01T00:00:00Z";
const DEFAULT_CONTEXT_WINDOW_SIZE = 200000;
const NGINX_404_HTML = `<html>
<head><title>404 Not Found</title></head>
<body>
<center><h1>404 Not Found</h1></center>
<hr><center>nginx</center>
</body>
</html>
`;

const HOP_BY_HOP_HEADERS = new Set([
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "trailers",
  "transfer-encoding",
  "upgrade",
]);

const REQUEST_HEADERS_TO_DROP = new Set([
  ...HOP_BY_HOP_HEADERS,
  "host",
  "content-length",
]);

const RESPONSE_HEADERS_TO_DROP = new Set([
  ...HOP_BY_HOP_HEADERS,
  "content-length",
]);

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = normalizeProxyPath(url.pathname);

    if (request.method === "POST" && path === "v1/messages") {
      return anthropicMessages(request, env);
    }

    if (request.method === "POST" && path === "v1/messages/count_tokens") {
      return anthropicCountTokens(request);
    }

    if (request.method === "GET" && path === "v1/models") {
      return listModels(request, env);
    }

    if (request.method === "GET" && (path === "health" || path === "healthz")) {
      return health(env);
    }

    return proxyRequest(request, env, path);
  },
};

async function anthropicMessages(request, env) {
  const payload = await readJsonRequest(request);
  if (!payload || Array.isArray(payload) || typeof payload !== "object") {
    return errorJson(400, "Invalid JSON payload");
  }

  const model = String(payload.model || "gpt-5.3-codex");
  const wantsStream = Boolean(payload.stream);
  logRequest(request, "v1/messages", {
    model,
    upstream_model: resolveUpstreamModel(model),
    stream: wantsStream,
    messages: Array.isArray(payload.messages) ? payload.messages.length : 0,
    tools: Array.isArray(payload.tools) ? payload.tools.length : 0,
  });

  const upstreamPayload = anthropicMessagesToResponsesPayload(payload);
  const upstreamUrl = buildUpstreamUrl(env, "v1/responses", "");
  if (upstreamUrl instanceof Response) {
    return upstreamUrl;
  }

  const upstreamHeaders = buildUpstreamHeaders(request, "v1/responses");
  upstreamHeaders.set("content-type", "application/json");

  const startedAt = performance.now();
  let upstreamResponse;
  try {
    upstreamResponse = await fetch(upstreamUrl, {
      method: "POST",
      headers: upstreamHeaders,
      body: JSON.stringify(upstreamPayload),
      redirect: "manual",
    });
  } catch (error) {
    return errorJson(502, `Upstream request failed: ${error.message}`);
  }

  const elapsedMs = Math.round(performance.now() - startedAt);
  logUpstreamResponse("v1/messages->v1/responses", upstreamResponse, elapsedMs);

  if (upstreamResponse.status >= 400) {
    return upstreamErrorResponse("v1/messages->v1/responses", upstreamResponse, elapsedMs);
  }

  if (wantsStream) {
    return new Response(streamFromAsyncIterable(anthropicStreamFromResponses(upstreamResponse, model)), {
      headers: { "content-type": "text/event-stream" },
    });
  }

  return anthropicMessageFromResponses(upstreamResponse, model);
}

async function anthropicCountTokens(request) {
  const payload = await readJsonRequest(request);
  if (!payload || Array.isArray(payload) || typeof payload !== "object") {
    return errorJson(400, "Invalid JSON payload");
  }

  logRequest(request, "v1/messages/count_tokens", {
    model: payload.model || "",
    messages: Array.isArray(payload.messages) ? payload.messages.length : 0,
  });

  let text = anthropicContentToText(payload.system || "");
  if (Array.isArray(payload.messages)) {
    for (const message of payload.messages) {
      if (message && typeof message === "object" && !Array.isArray(message)) {
        text += `\n${anthropicContentToText(message.content || "")}`;
      }
    }
  }

  return jsonResponse({ input_tokens: Math.max(1, Math.floor(text.length / 4)) });
}

async function listModels(request, env) {
  logRequest(request, "v1/models", {
    mode: isAnthropicRequest(request) ? "anthropic" : "openai",
  });

  const requestUrl = new URL(request.url);
  const upstreamUrl = buildUpstreamUrl(env, "v1/models", requestUrl.search);
  if (upstreamUrl instanceof Response) {
    return upstreamUrl;
  }

  const upstreamHeaders = buildUpstreamHeaders(request, "v1/models");
  const startedAt = performance.now();
  let upstreamResponse;
  try {
    upstreamResponse = await fetch(upstreamUrl, {
      method: "GET",
      headers: upstreamHeaders,
      redirect: "manual",
    });
  } catch (error) {
    return errorJson(502, `Upstream request failed: ${error.message}`);
  }

  const elapsedMs = Math.round(performance.now() - startedAt);
  logUpstreamResponse("v1/models", upstreamResponse, elapsedMs);

  if (upstreamResponse.status >= 400) {
    return upstreamErrorResponse("v1/models", upstreamResponse, elapsedMs);
  }

  const bodyText = await upstreamResponse.text();
  let payload;
  try {
    payload = JSON.parse(bodyText);
  } catch {
    return new Response(bodyForStatus(upstreamResponse.status, bodyText), {
      status: upstreamResponse.status,
      headers: filterResponseHeaders(upstreamResponse.headers),
    });
  }

  const rawModels = modelListFromPayload(payload);
  const contextSources = {};
  for (const rawModel of rawModels) {
    const contextSource = modelContextSource(rawModel);
    contextSources[contextSource] = (contextSources[contextSource] || 0) + 1;
  }
  console.log(
    "models_transform",
    JSON.stringify({
      mode: isAnthropicRequest(request) ? "anthropic" : "openai",
      count: rawModels.length,
      context_sources: contextSources,
    }),
  );

  if (isAnthropicRequest(request)) {
    return jsonResponse(toAnthropicModelsResponse(payload));
  }

  return jsonResponse(toOpenaiModelsResponse(payload));
}

async function proxyRequest(request, env, path) {
  if (!path) {
    return nginx404Response(request.method);
  }

  const requestUrl = new URL(request.url);
  logRequest(request, path, { query: Boolean(requestUrl.search) });

  const upstreamUrl = buildUpstreamUrl(env, path, requestUrl.search);
  if (upstreamUrl instanceof Response) {
    return upstreamUrl;
  }

  const init = {
    method: request.method,
    headers: buildUpstreamHeaders(request, path),
    redirect: "manual",
  };

  if (!["GET", "HEAD"].includes(request.method)) {
    init.body = request.body;
  }

  const startedAt = performance.now();
  let upstreamResponse;
  try {
    upstreamResponse = await fetch(upstreamUrl, init);
  } catch (error) {
    return errorJson(502, `Upstream request failed: ${error.message}`);
  }

  const elapsedMs = Math.round(performance.now() - startedAt);
  logUpstreamResponse(path, upstreamResponse, elapsedMs);

  if (upstreamResponse.status >= 400) {
    return upstreamErrorResponse(path, upstreamResponse, elapsedMs);
  }

  return new Response(responseBodyForStatus(upstreamResponse), {
    status: upstreamResponse.status,
    headers: filterResponseHeaders(upstreamResponse.headers),
  });
}

function health(env) {
  return jsonResponse({
    ok: true,
    has_upstream_base_url: Boolean(upstreamBaseUrl(env)),
    auth_source: "request_headers",
  });
}

function upstreamBaseUrl(env) {
  return String(env.CODEX_PROXY_UPSTREAM_BASE_URL || DEFAULT_UPSTREAM_BASE_URL).replace(/\/+$/, "");
}

function buildUpstreamUrl(env, path, query) {
  const base = upstreamBaseUrl(env);
  if (!base) {
    return errorJson(500, "CODEX_PROXY_UPSTREAM_BASE_URL is not configured");
  }

  const normalizedPath = normalizeProxyPath(path);
  return `${base}${normalizedPath ? `/${normalizedPath}` : ""}${query || ""}`;
}

function buildUpstreamHeaders(request, path) {
  const headers = new Headers();
  for (const [key, value] of request.headers) {
    if (REQUEST_HEADERS_TO_DROP.has(key.toLowerCase())) {
      continue;
    }
    headers.set(key, value);
  }

  const requestApiKey = extractRequestApiKey(request);
  if (requestApiKey) {
    headers.set("authorization", `Bearer ${requestApiKey}`);
    headers.set("x-api-key", requestApiKey);
  }

  if (isAnthropicCompatiblePath(path) && !headers.has("anthropic-version")) {
    headers.set("anthropic-version", DEFAULT_ANTHROPIC_VERSION);
  }

  if (!headers.has("user-agent")) {
    headers.set("user-agent", "codex-proxy/0.1.0");
  }

  return headers;
}

function filterResponseHeaders(sourceHeaders) {
  const headers = new Headers();
  for (const [key, value] of sourceHeaders) {
    if (RESPONSE_HEADERS_TO_DROP.has(key.toLowerCase())) {
      continue;
    }
    headers.set(key, value);
  }
  return headers;
}

function responseBodyForStatus(response) {
  return bodyForStatus(response.status, response.body);
}

function bodyForStatus(status, body) {
  return [204, 205, 304].includes(status) ? null : body;
}

async function upstreamErrorResponse(route, upstreamResponse, elapsedMs) {
  const body = await upstreamResponse.arrayBuffer();
  const preview = truncateLogValue(new TextDecoder().decode(body));
  console.warn(
    "upstream_error",
    JSON.stringify({
      route,
      status: upstreamResponse.status,
      elapsed_ms: elapsedMs,
      request_id: upstreamResponse.headers.get("x-request-id") || "",
      body: preview,
    }),
  );

  return new Response(body, {
    status: upstreamResponse.status,
    headers: filterResponseHeaders(upstreamResponse.headers),
  });
}

function normalizeProxyPath(path) {
  return String(path || "").replace(/^\/+|\/+$/g, "");
}

function extractBearer(headers) {
  const authorization = headers.get("authorization") || "";
  if (authorization.toLowerCase().startsWith("bearer ")) {
    return authorization.slice(7).trim();
  }
  return "";
}

function extractRequestApiKey(request) {
  return extractBearer(request.headers) || request.headers.get("x-api-key") || "";
}

function isAnthropicCompatiblePath(path) {
  const normalized = normalizeProxyPath(path);
  return normalized.startsWith("v1/messages") || normalized.startsWith("v1/complete");
}

function isAnthropicRequest(request) {
  return request.headers.has("anthropic-version");
}

function resolveUpstreamModel(model) {
  if (model.startsWith("anthropic/")) {
    return model.split("/", 2)[1];
  }
  return model;
}

function anthropicContentToText(content) {
  if (typeof content === "string") {
    return content;
  }

  if (!Array.isArray(content)) {
    return content == null ? "" : String(content);
  }

  const parts = [];
  for (const block of content) {
    if (typeof block === "string") {
      parts.push(block);
      continue;
    }

    if (!block || typeof block !== "object" || Array.isArray(block)) {
      continue;
    }

    if (block.type === "text") {
      parts.push(String(block.text || ""));
    } else if (block.type === "tool_use") {
      parts.push(`Tool use ${block.name || ""}: ${JSON.stringify(block.input || {})}`);
    } else if (block.type === "tool_result") {
      parts.push(`Tool result for ${block.tool_use_id || ""}: ${anthropicContentToText(block.content || "")}`);
    } else if (block.type === "image") {
      parts.push("[image]");
    }
  }

  return parts.filter(Boolean).join("\n");
}

function anthropicMessagesToResponsesPayload(payload) {
  const inputItems = [];
  if (payload.system) {
    inputItems.push({
      role: "system",
      content: anthropicContentToText(payload.system),
    });
  }

  if (Array.isArray(payload.messages)) {
    for (const message of payload.messages) {
      if (!message || typeof message !== "object" || Array.isArray(message)) {
        continue;
      }

      let role = message.role || "user";
      if (!["user", "assistant", "system", "developer"].includes(role)) {
        role = "user";
      }

      inputItems.push({
        role,
        content: anthropicContentToText(message.content || ""),
      });
    }
  }

  const responsesPayload = {
    model: resolveUpstreamModel(String(payload.model || "gpt-5.3-codex")),
    input: inputItems,
    stream: true,
  };

  if (Number.isInteger(payload.max_tokens) && payload.max_tokens > 0) {
    responsesPayload.max_output_tokens = payload.max_tokens;
  }

  if (typeof payload.temperature === "number") {
    responsesPayload.temperature = payload.temperature;
  }

  if (typeof payload.top_p === "number") {
    responsesPayload.top_p = payload.top_p;
  }

  if (Array.isArray(payload.tools) && payload.tools.length) {
    responsesPayload.tools = payload.tools
      .filter((tool) => tool && typeof tool === "object" && !Array.isArray(tool) && tool.name)
      .map((tool) => ({
        type: "function",
        name: tool.name,
        description: tool.description || "",
        parameters: tool.input_schema || { type: "object", properties: {} },
      }));
  }

  return responsesPayload;
}

function sseEvent(event, data) {
  const lines = [];
  if (event) {
    lines.push(`event: ${event}`);
  }

  const payload = typeof data === "string" ? data : JSON.stringify(data);
  for (const line of payload.split(/\r?\n/) || [""]) {
    lines.push(`data: ${line}`);
  }
  lines.push("");
  return `${lines.join("\n")}\n`;
}

async function* iterSseEvents(response) {
  if (!response.body) {
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let event = null;
  let dataLines = [];

  const processLine = async function* (rawLine) {
    const line = rawLine.endsWith("\r") ? rawLine.slice(0, -1) : rawLine;
    if (!line) {
      if (dataLines.length) {
        yield { event, data: dataLines.join("\n") };
      }
      event = null;
      dataLines = [];
      return;
    }

    if (line.startsWith(":")) {
      return;
    }
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });

    let newlineIndex;
    while ((newlineIndex = buffer.indexOf("\n")) !== -1) {
      const line = buffer.slice(0, newlineIndex);
      buffer = buffer.slice(newlineIndex + 1);
      yield* processLine(line);
    }
  }

  buffer += decoder.decode();
  if (buffer) {
    yield* processLine(buffer);
  }
  if (dataLines.length) {
    yield { event, data: dataLines.join("\n") };
  }
}

function extractResponseText(data) {
  if (typeof data.delta === "string") {
    return data.delta;
  }

  if (typeof data.text === "string") {
    return data.text;
  }

  if (data.response && typeof data.response === "object" && Array.isArray(data.response.output)) {
    const parts = [];
    for (const item of data.response.output) {
      if (!item || typeof item !== "object" || !Array.isArray(item.content)) {
        continue;
      }
      for (const content of item.content) {
        if (content && typeof content === "object" && typeof content.text === "string") {
          parts.push(content.text);
        }
      }
    }
    return parts.join("");
  }

  return "";
}

function safeInt(value, defaultValue = 0) {
  if (typeof value === "boolean") {
    return defaultValue;
  }
  if (Number.isInteger(value)) {
    return value;
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.trunc(value);
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? Math.trunc(parsed) : defaultValue;
  }
  return defaultValue;
}

function anthropicUsageFromResponseUsage(usage) {
  const normalizedUsage = usage && typeof usage === "object" && !Array.isArray(usage) ? usage : {};
  const totalInputTokens = safeInt(normalizedUsage.input_tokens ?? normalizedUsage.prompt_tokens, 0);
  const outputTokens = safeInt(normalizedUsage.output_tokens ?? normalizedUsage.completion_tokens, 0);
  const inputDetails = normalizedUsage.input_tokens_details || normalizedUsage.prompt_tokens_details || {};
  const cachedTokens = inputDetails && typeof inputDetails === "object" ? safeInt(inputDetails.cached_tokens, 0) : 0;

  return {
    input_tokens: Math.max(totalInputTokens - cachedTokens, 0),
    cache_creation_input_tokens: 0,
    cache_read_input_tokens: cachedTokens,
    output_tokens: outputTokens,
  };
}

function extractResponseUsage(data) {
  if (data.response && typeof data.response === "object") {
    return anthropicUsageFromResponseUsage(data.response.usage || {});
  }
  return anthropicUsageFromResponseUsage(data.usage || {});
}

function extractToolCallFromItem(item) {
  if (!item || typeof item !== "object" || Array.isArray(item)) {
    return null;
  }

  const itemType = item.type;
  if (!["function_call", "tool_call"].includes(itemType)) {
    return null;
  }

  const name = item.name || (item.function && item.function.name);
  if (!name) {
    return null;
  }

  let args = item.arguments;
  if (args == null && item.function) {
    args = item.function.arguments || "";
  }

  return {
    item_id: String(item.id || item.call_id || name),
    tool_use_id: String(item.call_id || item.id || `toolu_${Date.now()}`),
    name: String(name),
    arguments: typeof args === "object" && args !== null ? JSON.stringify(args) : String(args || ""),
  };
}

function extractCompletedToolCalls(data) {
  if (!data.response || typeof data.response !== "object" || !Array.isArray(data.response.output)) {
    return [];
  }

  const toolCalls = [];
  for (const item of data.response.output) {
    const toolCall = extractToolCallFromItem(item);
    if (toolCall) {
      toolCalls.push(toolCall);
    }
  }
  return toolCalls;
}

async function* anthropicStreamFromResponses(upstreamResponse, model) {
  const messageId = `msg_${Date.now()}`;
  let outputText = "";
  let textIndex = null;
  let nextContentIndex = 0;
  const toolBlocks = new Map();
  let toolUsed = false;
  let finalUsage = anthropicUsageFromResponseUsage({});

  async function* emitTextDelta(textDelta) {
    if (!textDelta) {
      return;
    }

    if (textIndex === null) {
      textIndex = nextContentIndex;
      nextContentIndex += 1;
      yield sseEvent("content_block_start", {
        type: "content_block_start",
        index: textIndex,
        content_block: { type: "text", text: "" },
      });
    }

    outputText += textDelta;
    yield sseEvent("content_block_delta", {
      type: "content_block_delta",
      index: textIndex,
      delta: { type: "text_delta", text: textDelta },
    });
  }

  async function* startToolBlock(toolCall) {
    const itemId = toolCall.item_id;
    if (!toolBlocks.has(itemId)) {
      toolBlocks.set(itemId, {
        index: nextContentIndex,
        tool_use_id: toolCall.tool_use_id,
        name: toolCall.name,
        arguments: "",
        started: false,
        stopped: false,
      });
      nextContentIndex += 1;
    }

    const block = toolBlocks.get(itemId);
    if (!block.started) {
      block.started = true;
      toolUsed = true;
      yield sseEvent("content_block_start", {
        type: "content_block_start",
        index: block.index,
        content_block: {
          type: "tool_use",
          id: block.tool_use_id,
          name: block.name,
          input: {},
        },
      });
    }
  }

  async function* emitToolArguments(toolCall, argumentsDelta) {
    yield* startToolBlock(toolCall);

    if (!argumentsDelta) {
      return;
    }

    const block = toolBlocks.get(toolCall.item_id);
    block.arguments += argumentsDelta;
    yield sseEvent("content_block_delta", {
      type: "content_block_delta",
      index: block.index,
      delta: { type: "input_json_delta", partial_json: argumentsDelta },
    });
  }

  async function* stopToolBlock(toolCall) {
    yield* startToolBlock(toolCall);

    const block = toolBlocks.get(toolCall.item_id);
    if (!block.arguments && toolCall.arguments) {
      yield* emitToolArguments(toolCall, toolCall.arguments);
    }

    if (!block.stopped) {
      block.stopped = true;
      yield sseEvent("content_block_stop", {
        type: "content_block_stop",
        index: block.index,
      });
    }
  }

  yield sseEvent("message_start", {
    type: "message_start",
    message: {
      id: messageId,
      type: "message",
      role: "assistant",
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: finalUsage,
    },
  });

  for await (const { event, data: rawData } of iterSseEvents(upstreamResponse)) {
    if (rawData === "[DONE]") {
      continue;
    }

    let data;
    try {
      data = JSON.parse(rawData);
    } catch {
      continue;
    }

    const eventType = data.type || event;
    if (eventType === "response.output_text.delta") {
      yield* emitTextDelta(extractResponseText(data));
    } else if (eventType === "response.output_text.done") {
      const doneText = extractResponseText(data);
      if (doneText && doneText.startsWith(outputText)) {
        yield* emitTextDelta(doneText.slice(outputText.length));
      } else if (doneText && !outputText) {
        yield* emitTextDelta(doneText);
      }
    } else if (eventType === "response.output_item.added") {
      const toolCall = extractToolCallFromItem(data.item);
      if (toolCall) {
        yield* startToolBlock(toolCall);
      }
    } else if (eventType === "response.function_call_arguments.delta") {
      const itemId = String(data.item_id || data.output_item_id || data.call_id || "");
      const block = toolBlocks.get(itemId);
      if (block) {
        yield* emitToolArguments(
          {
            item_id: itemId,
            tool_use_id: String(block.tool_use_id),
            name: String(block.name),
            arguments: "",
          },
          String(data.delta || ""),
        );
      }
    } else if (eventType === "response.function_call_arguments.done") {
      const itemId = String(data.item_id || data.output_item_id || data.call_id || "");
      const block = toolBlocks.get(itemId);
      if (block) {
        yield* stopToolBlock({
          item_id: itemId,
          tool_use_id: String(block.tool_use_id),
          name: String(block.name),
          arguments: String(data.arguments || ""),
        });
      }
    } else if (eventType === "response.output_item.done") {
      const toolCall = extractToolCallFromItem(data.item);
      if (toolCall) {
        yield* stopToolBlock(toolCall);
      }
    } else if (eventType === "response.completed") {
      const completedText = extractResponseText(data);
      if (completedText && completedText.startsWith(outputText)) {
        yield* emitTextDelta(completedText.slice(outputText.length));
      } else if (completedText && !outputText) {
        yield* emitTextDelta(completedText);
      }

      for (const toolCall of extractCompletedToolCalls(data)) {
        yield* stopToolBlock(toolCall);
      }

      finalUsage = extractResponseUsage(data);
    } else if (["response.failed", "response.incomplete", "error"].includes(eventType)) {
      const error = data.error || data;
      const message = error && typeof error === "object" ? error.message || "Upstream response failed" : String(error);
      yield sseEvent("error", {
        type: "error",
        error: { type: "api_error", message },
      });
      return;
    }
  }

  if (textIndex !== null) {
    yield sseEvent("content_block_stop", {
      type: "content_block_stop",
      index: textIndex,
    });
  }

  for (const block of toolBlocks.values()) {
    if (!block.stopped) {
      block.stopped = true;
      yield sseEvent("content_block_stop", {
        type: "content_block_stop",
        index: block.index,
      });
    }
  }

  if (textIndex === null && !toolUsed) {
    yield sseEvent("content_block_start", {
      type: "content_block_start",
      index: 0,
      content_block: { type: "text", text: "" },
    });
    yield sseEvent("content_block_stop", {
      type: "content_block_stop",
      index: 0,
    });
  }

  yield sseEvent("message_delta", {
    type: "message_delta",
    delta: {
      stop_reason: toolUsed ? "tool_use" : "end_turn",
      stop_sequence: null,
    },
    usage: finalUsage,
  });
  yield sseEvent("message_stop", { type: "message_stop" });
}

async function anthropicMessageFromResponses(upstreamResponse, model) {
  let outputText = "";
  let usage = anthropicUsageFromResponseUsage({});
  const toolCalls = [];

  for await (const { event, data: rawData } of iterSseEvents(upstreamResponse)) {
    if (rawData === "[DONE]") {
      continue;
    }

    let data;
    try {
      data = JSON.parse(rawData);
    } catch {
      continue;
    }

    const eventType = data.type || event;
    if (eventType === "response.output_text.delta") {
      outputText += extractResponseText(data);
    } else if (eventType === "response.output_text.done") {
      const doneText = extractResponseText(data);
      if (doneText && doneText.startsWith(outputText)) {
        outputText += doneText.slice(outputText.length);
      } else if (doneText && !outputText) {
        outputText = doneText;
      }
    } else if (eventType === "response.output_item.done") {
      const toolCall = extractToolCallFromItem(data.item);
      if (toolCall) {
        toolCalls.push(toolCall);
      }
    } else if (eventType === "response.completed") {
      const completedText = extractResponseText(data);
      if (completedText && completedText.startsWith(outputText)) {
        outputText += completedText.slice(outputText.length);
      } else if (completedText && !outputText) {
        outputText = completedText;
      }
      usage = extractResponseUsage(data);
      toolCalls.push(...extractCompletedToolCalls(data));
    }
  }

  const content = [];
  if (outputText) {
    content.push({ type: "text", text: outputText });
  }

  const seenToolIds = new Set();
  for (const toolCall of toolCalls) {
    if (seenToolIds.has(toolCall.tool_use_id)) {
      continue;
    }
    seenToolIds.add(toolCall.tool_use_id);

    let toolInput;
    try {
      toolInput = JSON.parse(toolCall.arguments || "{}");
    } catch {
      toolInput = { arguments: toolCall.arguments || "" };
    }

    content.push({
      type: "tool_use",
      id: toolCall.tool_use_id,
      name: toolCall.name,
      input: toolInput,
    });
  }

  return jsonResponse({
    id: `msg_${Date.now()}`,
    type: "message",
    role: "assistant",
    model,
    content: content.length ? content : [{ type: "text", text: "" }],
    stop_reason: toolCalls.length ? "tool_use" : "end_turn",
    stop_sequence: null,
    usage,
  });
}

function modelContextSource(rawModel) {
  if (rawModel && typeof rawModel === "object" && !Array.isArray(rawModel)) {
    for (const key of [
      "context_window_size",
      "context_window",
      "context_length",
      "max_context_window",
      "max_context_tokens",
      "max_input_tokens",
    ]) {
      if (safeInt(rawModel[key], 0) > 0) {
        return key;
      }
    }
  }
  return "default";
}

function modelIdFromRaw(rawModel) {
  if (typeof rawModel === "string") {
    return rawModel;
  }
  if (!rawModel || typeof rawModel !== "object" || Array.isArray(rawModel)) {
    return "";
  }
  return String(rawModel.id || rawModel.slug || rawModel.name || "");
}

function displayNameFromRaw(rawModel, modelId) {
  if (rawModel && typeof rawModel === "object" && !Array.isArray(rawModel)) {
    const value = rawModel.display_name || rawModel.name || rawModel.id || rawModel.slug;
    if (value) {
      return String(value);
    }
  }
  return modelId;
}

function createdAtFromRaw(rawModel) {
  if (rawModel && typeof rawModel === "object" && !Array.isArray(rawModel)) {
    if (typeof rawModel.created_at === "string" && rawModel.created_at) {
      return rawModel.created_at;
    }
    if (typeof rawModel.created === "number") {
      return new Date(rawModel.created * 1000).toISOString().replace(".000Z", "Z");
    }
  }
  return DEFAULT_MODEL_CREATED_AT;
}

function contextWindowFromRaw(rawModel) {
  if (rawModel && typeof rawModel === "object" && !Array.isArray(rawModel)) {
    for (const key of [
      "context_window_size",
      "context_window",
      "context_length",
      "max_context_window",
      "max_context_tokens",
      "max_input_tokens",
    ]) {
      const value = safeInt(rawModel[key], 0);
      if (value > 0) {
        return value;
      }
    }
  }
  return DEFAULT_CONTEXT_WINDOW_SIZE;
}

function modelListFromPayload(payload) {
  if (payload && typeof payload === "object" && !Array.isArray(payload)) {
    if (Array.isArray(payload.data)) {
      return payload.data;
    }
    if (Array.isArray(payload.models)) {
      return payload.models;
    }
  }
  if (Array.isArray(payload)) {
    return payload;
  }
  return [];
}

function toAnthropicModelsResponse(payload) {
  const data = [];
  for (const rawModel of modelListFromPayload(payload)) {
    const upstreamModelId = modelIdFromRaw(rawModel);
    if (!upstreamModelId) {
      continue;
    }

    const contextWindowSize = contextWindowFromRaw(rawModel);
    data.push({
      type: "model",
      id: upstreamModelId,
      display_name: displayNameFromRaw(rawModel, upstreamModelId),
      created_at: createdAtFromRaw(rawModel),
      max_input_tokens: contextWindowSize,
      context_window_size: contextWindowSize,
    });
  }

  return {
    data,
    has_more: false,
    first_id: data.length ? data[0].id : null,
    last_id: data.length ? data[data.length - 1].id : null,
  };
}

function toOpenaiModelsResponse(payload) {
  const data = [];
  for (const rawModel of modelListFromPayload(payload)) {
    const modelId = modelIdFromRaw(rawModel);
    if (!modelId) {
      continue;
    }

    const createdAt = createdAtFromRaw(rawModel);
    const parsed = Date.parse(createdAt);
    data.push({
      id: modelId,
      object: "model",
      created: Number.isFinite(parsed) ? Math.floor(parsed / 1000) : Math.floor(Date.now() / 1000),
      owned_by: "upstream",
      display_name: displayNameFromRaw(rawModel, modelId),
    });
  }

  return { object: "list", data };
}

function streamFromAsyncIterable(iterable) {
  const encoder = new TextEncoder();
  const iterator = iterable[Symbol.asyncIterator]();

  return new ReadableStream({
    async pull(controller) {
      const { value, done } = await iterator.next();
      if (done) {
        controller.close();
        return;
      }
      controller.enqueue(typeof value === "string" ? encoder.encode(value) : value);
    },
    async cancel() {
      if (typeof iterator.return === "function") {
        await iterator.return();
      }
    },
  });
}

async function readJsonRequest(request) {
  try {
    return await request.json();
  } catch {
    return null;
  }
}

function jsonResponse(payload, status = 200) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function errorJson(status, detail) {
  return jsonResponse({ detail }, status);
}

function nginx404Response(method) {
  return new Response(method === "HEAD" ? null : NGINX_404_HTML, {
    status: 404,
    headers: {
      "content-type": "text/html",
    },
  });
}

function truncateLogValue(value, limit = 1200) {
  return value.length <= limit ? value : `${value.slice(0, limit)}...<truncated>`;
}

function requestAuthSource(request) {
  if (extractBearer(request.headers)) {
    return "authorization";
  }
  if (request.headers.get("x-api-key")) {
    return "x-api-key";
  }
  return "missing";
}

function logRequest(request, route, fields = {}) {
  console.log(
    "request",
    JSON.stringify({
      route,
      method: request.method,
      auth: requestAuthSource(request),
      ...fields,
    }),
  );
}

function logUpstreamResponse(route, upstreamResponse, elapsedMs) {
  console.log(
    "upstream",
    JSON.stringify({
      route,
      status: upstreamResponse.status,
      elapsed_ms: elapsedMs,
      request_id: upstreamResponse.headers.get("x-request-id") || "",
      content_type: upstreamResponse.headers.get("content-type") || "",
    }),
  );
}
