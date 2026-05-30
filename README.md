# Codex API Proxy

Python 3 proxy for OpenAI-compatible clients, OpenAI Codex CLI style `/v1/*`
routes, and Claude Code / Anthropic-compatible `/v1/messages*` routes.

The proxy forwards requests to the upstream configured by
`CODEX_PROXY_UPSTREAM_BASE_URL`. Local client authentication is disabled by
default, so tools can point at this proxy without sending the upstream key. The
proxy injects the upstream key from `CODEX_PROXY_UPSTREAM_API_KEY`.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `CODEX_PROXY_UPSTREAM_BASE_URL` and
`CODEX_PROXY_UPSTREAM_API_KEY`.

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Docker Compose

```bash
cp .env.example .env
# edit .env and set CODEX_PROXY_UPSTREAM_API_KEY
docker compose up -d --build
```

The compose setup follows the reference project layout:

- `app` runs `uvicorn` on `APP_PORT`, default `8000`
- `nginx` proxies to the app with streaming buffering disabled
- public nginx port is `NGINX_PORT`, default `4002`
- both services use `network_mode: host`

After deployment, use:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:4002/v1
export ANTHROPIC_BASE_URL=http://127.0.0.1:4002
```

## OpenAI / Codex CLI

Use this proxy as the OpenAI base URL:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=anything
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
export ANTHROPIC_AUTH_TOKEN=anything
export ANTHROPIC_API_KEY=anything
```

The proxy forwards `/v1/messages`, `/v1/messages/count_tokens`, `/v1/models`,
and related Anthropic-compatible routes. It sends both `Authorization` and
`x-api-key` upstream, plus a default `anthropic-version` header for Anthropic
message routes.

## Optional Local Auth

To require callers to authenticate to the proxy itself:

```bash
CODEX_PROXY_LOCAL_API_KEY=local-proxy-token
CODEX_PROXY_ALLOW_UNAUTHENTICATED=false
```

Clients may then send either:

```text
Authorization: Bearer local-proxy-token
```

or:

```text
x-api-key: local-proxy-token
```
