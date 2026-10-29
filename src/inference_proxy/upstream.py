"""Upstream HTTP helpers: payloads, headers, and error classification."""

from __future__ import annotations

import json

import httpx

from .context import looks_like_context_overflow
from .pool import AUTH, OVERFLOW, QUOTA, REQUEST, SERVER, UNSUPPORTED

def chat_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


def build_payload(
    body: dict,
    model: str,
    provider: object | None = None,
    tier_policy: object | None = None,
) -> dict:
    """Build the upstream payload.

    Precedence per field: explicit client value > tier policy > provider
    defaults. The provider ``max_tokens_cap`` is an absolute clamp applied
    last, regardless of who asked for what.
    """
    payload = dict(body)
    payload["model"] = model
    payload.pop("inference_proxy", None)

    defaults: dict = {}
    if provider is not None and getattr(provider, "extra_body", None):
        defaults = deep_merge(defaults, provider.extra_body)
    if tier_policy is not None and getattr(tier_policy, "extra_body", None):
        defaults = deep_merge(defaults, tier_policy.extra_body)
    payload = deep_merge(defaults, payload)

    if tier_policy is not None:
        default_max = getattr(tier_policy, "max_tokens", None)
        if default_max is not None and payload.get("max_tokens") is None:
            payload["max_tokens"] = default_max
    cap = getattr(provider, "max_tokens_cap", None) if provider is not None else None
    if cap is not None:
        current = payload.get("max_tokens")
        if current is None or current > cap:
            payload["max_tokens"] = cap
    return payload


def deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def build_headers(api_key: str | None) -> dict[str, str]:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def read_error_detail(status: int, body: bytes) -> str:
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            error = parsed.get("error", parsed.get("message"))
            if isinstance(error, dict):
                error = error.get("message")
            if error:
                return str(error)[:300]
    except (ValueError, AttributeError):
        pass
    return body.decode("utf-8", "replace")[:300] if body else f"HTTP {status}"


_MEDIA_WORDS = ("image", "video", "audio", "file", "vision", "multimodal", "modality")
_REJECT_PHRASES = (
    "not support",
    "unsupported",
    "does not support",
    "cannot handle",
    "invalid content",
    "content type",
)


def looks_like_media_rejection(detail: str) -> bool:
    """Heuristic: an upstream 400 that names a media type it won't accept."""
    detail = detail.lower()
    return (
        any(word in detail for word in _MEDIA_WORDS)
        and any(phrase in detail for phrase in _REJECT_PHRASES)
    )


def classify_upstream_error(status: int, body: bytes) -> str:
    if status in (401, 403):
        return AUTH
    if status == 413:
        # "content too large" from an inference API is a context overflow
        return OVERFLOW
    if status in (402, 429):
        return QUOTA
    if status >= 500 or status in (408, 425):
        return SERVER
    if 400 <= status < 500:
        detail = read_error_detail(status, body).lower()
        # overflow must win over the quota markers: "context length
        # exceeded" is a request-size problem, not an account-health one
        if looks_like_context_overflow(detail):
            return OVERFLOW
        quota_markers = ("quota", "insufficient", "credit", "exceeded", "balance", "limit reached")
        if any(marker in detail for marker in quota_markers):
            return QUOTA
        if looks_like_media_rejection(detail):
            # provider can't serve this modality; caller should fail over
            return UNSUPPORTED
        return REQUEST
    return SERVER
