# Codex API Proxy

Python 3 proxy for OpenAI-compatible clients, OpenAI Codex CLI style `/v1/*`
routes, and Claude Code / Anthropic-compatible `/v1/messages*` routes.

The proxy forwards requests to the upstream configured by
`CODEX_PROXY_UPSTREAM_BASE_URL`. API keys are read from each incoming request,
then forwarded upstream as `Authorization: Bearer ...` and `x-api-key`.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `CODEX_PROXY_UPSTREAM_BASE_URL`.

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Logs are written to stdout. Set `CODEX_PROXY_LOG_LEVEL=DEBUG` or
`CODEX_PROXY_LOG_LEVEL=INFO` to adjust verbosity. Logs include request route,
auth header source, upstream status, request ID, model transform counts, and
truncated upstream error bodies; API keys are not logged.

Optional model controls:

- `CODEX_PROXY_MODEL_MAP`: JSON object for direct upstream model remapping, for
  example `{"gpt-5.3-codex-spark":"gpt-5.4"}`.
- `CODEX_PROXY_RETRY_MODELS`: comma-separated upstream model fallback list used
  only after retryable upstream failures, for example `gpt-5.4,gpt-5.5`.

## Docker Compose

```bash
cp .env.example .env
# edit .env and set CODEX_PROXY_UPSTREAM_BASE_URL
docker compose up -d --build
```

The compose setup follows the reference project layout:

- `app` runs `uvicorn` on `APP_PORT`, default `8000`
- `nginx` proxies to the app with streaming buffering disabled
- public nginx port is `NGINX_PORT`, default `4002`
- services use Docker's default bridge network

After deployment, use:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:4002/v1
export ANTHROPIC_BASE_URL=http://127.0.0.1:4002
```

## OpenAI / Codex CLI

Use this proxy as the OpenAI base URL:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=your-upstream-api-key
```

The proxy forwards paths such as:

- `/v1/models`
- `/v1/responses`
- `/v1/chat/completions`
- `/v1/completions`
- `/v1/embeddings`
- any other OpenAI-compatible `/v1/*` route

## Claude Code / Anthropic-compatible clients

Use this proxy as the Anthropic base URL:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8000
export ANTHROPIC_AUTH_TOKEN=your-upstream-api-key
export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1
```

The proxy forwards `/v1/messages`, `/v1/messages/count_tokens`, `/v1/models`,
and related Anthropic-compatible routes. It adds a default `anthropic-version`
header for Anthropic message routes.

For Claude Code model discovery, `/v1/models` returns Anthropic-native model
metadata with upstream model IDs when the request includes
`anthropic-version`.
