# inference-proxy

An OpenAI-compatible proxy that pre-analyses task difficulty on your local LM
and routes accordingly:

- **trivial** → local Qwen 3.5 (llama.cpp / LM Studio)
- **moderate** → Z.ai, rotating between multiple accounts
- **complex** → Z.ai stronger model
- **hardcore** → OpenRouter (DeepSeek) / Kilo Gateway

Each provider holds a pool of accounts. When an account is rate-limited or out
of quota (429/402, or quota-sounding error bodies), it is parked with a
cooldown and the next account — or the next provider, or a neighbouring tier —
takes over.

## Quick start

```bash
uv sync
cp config.example.yaml config.yaml   # fill in keys, or export the env vars
uv run inference-proxy               # reads ./config.yaml, or set INFERENCE_PROXY_CONFIG
```

## Client usage

Point any OpenAI-compatible client at `http://127.0.0.1:8480/v1`. The `model`
field picks the routing mode:

| model value        | behaviour                                             |
|--------------------|-------------------------------------------------------|
| `auto` (default)   | local judge scores 1–10, tier chosen from thresholds   |
| `trivial`/`moderate`/`complex`/`hardcore` | force a tier, no judging                        |
| `zai`              | pin a provider (its default model)                     |
| `zai:glm-4.7`      | pin a provider and exact model                         |

Responses carry routing metadata:

- headers: `X-Proxy-Tier`, `X-Proxy-Provider`, `X-Proxy-Account`,
  `X-Proxy-Difficulty`, `X-Proxy-Difficulty-Method`
- non-streaming bodies also include an `inference_proxy` object with the same.

Streaming (SSE) is passed through untouched.

## Routing details

- Difficulty judging asks the local model for a single 1–10 score; if the
  judge is down or unparseable, a heuristic score is used instead.
- Account selection prefers the least-busy, least-recently-used healthy
  account.
- Failure handling: quota/auth errors park an account for
  `quota_cooldown_seconds` / `auth_failure_cooldown_seconds`; server and
  network errors back off exponentially (5 s → 60 s cap). Client errors (other
  4xx) are relayed verbatim without retrying.
- When a tier's accounts are all cooling down, the proxy walks to lower tiers
  first, then higher ones (`routing.fallback_to_other_tiers`).
- **Capabilities gate media and reasoning.** Each provider declares what it
  accepts: `capabilities: [text, image, video, audio, file, reasoning]`.
  Requests containing media parts (`image_url`, `video_url`, `input_audio`,
  `file`) or explicitly requesting reasoning (`reasoning_effort`, `reasoning`,
  `enable_thinking`, `thinking`) are only routed to providers that declare the
  matching capability — the difficulty tier is chosen as usual, then
  incapable providers are skipped, sliding across tiers if needed. If no
  configured provider qualifies, the proxy answers 400 instead of forwarding
  a doomed request. Pinning (`zai:glm-4.7`) bypasses the filter: pinned means
  pinned, and an upstream rejection is relayed. As a safety net, if a
  provider rejects a request as unsupported media anyway (e.g. the configured
  default model lacks vision despite the provider claiming it), the proxy
  fails over to the next capable provider rather than relaying the error —
  except when it was the last option left.
- `GET /admin/pools` shows per-account health, cooldowns, and counters.
- Pool cooldowns persist to `server.state_file` across restarts.

## Admin UI

Open `http://127.0.0.1:8480/admin` (also `/`) for a built-in dashboard:

- edit routing thresholds, the difficulty classifier, tier assignments and
  provider accounts — changes hot-apply without a restart
- add/remove API keys per provider account
- watch live pool status (cooldowns, failure streaks, counters) and reset
  stuck accounts
- fire a test request and see which tier/provider/account it routed to

UI edits persist to `ui-settings.json` (configurable via
`server.settings_file`, `null` disables) as an overlay that is deep-merged
over your YAML config on startup — the YAML file stays the base of truth and
is never modified. The overlay stores keys in plaintext, so keep it out of
version control (already in `.gitignore`).

The REST API behind the UI, if you want to script it:

| endpoint | method | purpose |
|---|---|---|
| `/admin/api/settings` | GET | effective settings, keys masked |
| `/admin/api/settings` | PUT | apply + persist settings |
| `/admin/api/pools/{provider}/accounts/{index}/reset` | POST | clear an account's cooldown |
| `/admin/pools` | GET | raw pool snapshot |

Keys are returned masked (`xxx••••abcd`); sending a masked value back means
"keep the current key". If `server.api_key` is set, admin endpoints (and the
UI's requests) require it as a Bearer token.

## Spending guardrails

Reasoning models (e.g. GLM-5.3 with thinking enabled) can burn large token
budgets before writing any visible answer. The proxy has a precedence model:

**explicit client value > tier policy > provider default**

- **Tier policies** (`tier_policies` per trivial/moderate/complex/hardcore, editable in the
  UI under each tier): a `max_tokens` default (injected only when the client
  sends none) and `extra_body` defaults (deep-merged field-by-field only
  where the client stayed silent — so a client asking for a specific
  reasoning level always gets it).
- **Provider** `extra_body`: endpoint-level defaults, same fill-in-the-gaps
  rule, overridden by tier policy.
- **Provider** `max_tokens_cap`: an opt-in absolute clamp applied last no
  matter who asked for what. Off by default; use it only when you want a
  hard ceiling on a provider.

Example: bound routine work but leave hard tasks to the client:

```yaml
tier_policies:
  trivial:
    max_tokens: 2048
    extra_body: {"thinking": {"type": "disabled"}}
  moderate:
    max_tokens: 8192
  hardcore: {}      # client decides
```

## Providers are generic

Provider names are arbitrary labels — `local`, `zai` and friends in the
example config are just conventions. Add, edit and delete providers freely in
the admin UI (or by editing `config.yaml`); each is an OpenAI-compatible
endpoint with any number of rotating accounts, assignable to any tier, and
pinnable per-request as `name` or `name:model`. Deleting a provider in the UI
unassigns it and persists the deletion across restarts; a tier can never be
left empty.

## Configuration

See `config.example.yaml`. All `${VAR}` / `${VAR:-default}` references expand
from the environment, so keys never need to live in the file.

## Development

```bash
uv run pytest          # tests
uv run uvicorn inference_proxy.main:create_app --factory --reload
```