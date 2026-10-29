"""Provider context-window discovery: probing, overflow-error learning, caching.

Context sizes are discovered in tokens (what servers report) and converted to
characters with a conservative divisor, since the proxy measures prompts in
chars. Discovery order: manual config override (classifier.context_chars),
LM Studio REST probe, then overflow-error learning at runtime.
"""

from __future__ import annotations

import re

import httpx

# Conservative chars-per-token: overestimates tokens for prose (~4/token),
# underestimates capacity for dense code/CJK, so the oversize guard errs safe.
CHARS_PER_TOKEN = 3

# LM Studio exposes the configured CTX of loaded instances on its native REST
# API, not on the OpenAI-compatible /v1 surface.
_PROBE_PATHS = ("/api/v1/models", "/api/v0/models")
_PROBE_TIMEOUT_SECONDS = 5.0


def looks_like_context_overflow(detail: str) -> bool:
    """Heuristic: an upstream rejection naming a context/token limit."""
    detail = detail.lower()
    markers = (
        "context length",
        "context window",
        "context size",
        "maximum context",
        "available context",
        "prompt is too long",
        "too many tokens",
        "input length exceed",
        "reduce the length",
    )
    return any(marker in detail for marker in markers)


# First group that matches wins; every alternative has exactly one capture.
_OVERFLOW_NUMBER_RE = re.compile(
    r"maximum context length is (\d+)"
    r"|exceed(?:s|ed)?\s+context window\s*\((\d+)\)"
    r"|context window\s*(?:is|of)\s*(\d+)"
    r"|context length\s*(?:is|of)\s*(\d+)"
    r"|maximum (?:of\s*)?(\d+) tokens"
    r"|\d+\s*tokens?\s*>\s*(\d+)",
    re.IGNORECASE,
)


def parse_overflow_tokens(text: str) -> int | None:
    """Extract the context limit (tokens) from an overflow error message."""
    match = _OVERFLOW_NUMBER_RE.search(text)
    if match is None:
        return None
    for group in match.groups():
        if group is not None:
            return int(group)
    return None


def _model_entry(payload: object, model: str) -> dict | None:
    if not isinstance(payload, dict):
        return None
    models = payload.get("models") or payload.get("data")
    if not isinstance(models, list):
        return None
    exact, fuzzy = [], []
    for entry in models:
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("key") or entry.get("id") or "")
        instances = entry.get("loaded_instances")
        if key == model or (
            isinstance(instances, list)
            and any(isinstance(i, dict) and i.get("id") == model for i in instances)
        ):
            exact.append(entry)
        elif key and (key.endswith("/" + model) or model.endswith("/" + key)):
            fuzzy.append(entry)
    candidates = exact or fuzzy
    if not candidates:
        return None
    # A loaded instance reports its real configured CTX; otherwise fall back
    # to the model file's maximum.
    return next((e for e in candidates if e.get("loaded_instances")), candidates[0])


def find_context_tokens(payload: object, model: str) -> int | None:
    """Pull the enforced context size (tokens) out of an LM Studio models payload."""
    entry = _model_entry(payload, model)
    if entry is None:
        return None
    for instance in entry.get("loaded_instances") or []:
        if not isinstance(instance, dict):
            continue
        config = instance.get("config")
        if isinstance(config, dict):
            tokens = config.get("context_length")
            if isinstance(tokens, int) and tokens > 0:
                return tokens
    tokens = entry.get("max_context_length")
    return tokens if isinstance(tokens, int) and tokens > 0 else None


class ContextRegistry:
    """Known context windows in tokens, keyed by (base_url, model)."""

    def __init__(self) -> None:
        self._tokens: dict[tuple[str, str], int] = {}

    @staticmethod
    def key(base_url: str, model: str) -> tuple[str, str]:
        return (base_url.rstrip("/"), model)

    def known_tokens(self, base_url: str, model: str) -> int | None:
        return self._tokens.get(self.key(base_url, model))

    def learn_tokens(self, base_url: str, model: str, tokens: int) -> None:
        if isinstance(tokens, int) and tokens > 0:
            self._tokens[self.key(base_url, model)] = tokens

    def learn_from_error(self, base_url: str, model: str, body: bytes) -> bool:
        """Learn the limit from an overflow error body; returns True on success."""
        text = body.decode("utf-8", "replace")
        if not looks_like_context_overflow(text):
            return False
        tokens = parse_overflow_tokens(text)
        if tokens is None:
            return False
        self.learn_tokens(base_url, model, tokens)
        return True


async def probe_model_context(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    api_key: str | None = None,
    registry: ContextRegistry | None = None,
) -> int | None:
    """Best-effort probe of an LM Studio server for a model's context size."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    for path in _PROBE_PATHS:
        try:
            response = await client.get(
                root + path,
                headers=headers,
                timeout=httpx.Timeout(_PROBE_TIMEOUT_SECONDS),
            )
            if response.status_code != 200:
                continue
            tokens = find_context_tokens(response.json(), model)
        except (httpx.HTTPError, ValueError):
            continue
        if tokens:
            if registry is not None:
                registry.learn_tokens(base_url, model, tokens)
            return tokens
    return None
