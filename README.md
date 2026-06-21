# Codex API Proxy

Cloudflare Worker proxy for OpenAI-compatible clients, OpenAI Codex CLI style
`/v1/*` routes, and Claude Code / Anthropic-compatible `/v1/messages*` routes.

The proxy forwards requests to the upstream configured by
`CODEX_PROXY_UPSTREAM_BASE_URL`. API keys are read from each incoming request,
then forwarded upstream as `Authorization: Bearer ...` and `x-api-key`.

This project now targets Cloudflare Workers only. Docker, nginx, and Python
runtime deployment paths have been removed.

## Setup

```bash
pnpm install
```

Configure the upstream base URL for the Worker:

```bash
pnpm exec wrangler secret put CODEX_PROXY_UPSTREAM_BASE_URL
```

Set the secret value to the upstream API base URL, for example
`https://api.example.com`.

## Deploy

```bash
pnpm run deploy
```

To validate the Worker bundle without deploying:

```bash
pnpm run deploy:dry-run
```

## Local Dev

Copy the example local vars file and fill in the upstream URL:

```bash
cp .dev.vars.example .dev.vars
pnpm dev
```

After deployment, use your Worker URL:

```bash
export OPENAI_BASE_URL=https://codex-proxy.<account>.workers.dev/v1
export ANTHROPIC_BASE_URL=https://codex-proxy.<account>.workers.dev
```

## OpenAI / Codex CLI

Use this proxy as the OpenAI base URL:

```bash
export OPENAI_BASE_URL=https://codex-proxy.<account>.workers.dev/v1
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
export ANTHROPIC_BASE_URL=https://codex-proxy.<account>.workers.dev
export ANTHROPIC_AUTH_TOKEN=your-upstream-api-key
export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1
```

The proxy forwards `/v1/messages`, `/v1/messages/count_tokens`, `/v1/models`,
and related Anthropic-compatible routes. It adds a default `anthropic-version`
header for Anthropic message routes.

For Claude Code model discovery, `/v1/models` returns Anthropic-native model
metadata with upstream model IDs when the request includes
`anthropic-version`.

## Health Checks

```bash
curl https://codex-proxy.<account>.workers.dev/health
curl https://codex-proxy.<account>.workers.dev/healthz
```
