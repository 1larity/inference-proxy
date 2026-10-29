"""Capability detection: what a request needs, determined without an LLM.

Media is detected from message content-part types (OpenAI-compatible plus
GLM's ``video_url``); reasoning is detected from explicit reasoning knobs in
the request body. Detection is deterministic — a request either contains
media or it does not — so routing can hard-filter providers by capability
instead of guessing from difficulty.
"""

from __future__ import annotations

CAPABILITIES = ("text", "image", "video", "audio", "file", "reasoning")

# Content-part type -> capability it requires.
PART_CAPABILITIES = {
    "image_url": "image",
    "video_url": "video",
    "input_audio": "audio",
    "file": "file",
}

_TEXT_ONLY = frozenset({"text"})
_OFF_VALUES = frozenset({"", "none", "off", "disabled", "false"})


def part_capability(part: object) -> str | None:
    if isinstance(part, dict):
        return PART_CAPABILITIES.get(str(part.get("type", "")))
    return None


def _flag_on(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        kind = value.get("type")
        if isinstance(kind, str):
            return kind.strip().lower() not in _OFF_VALUES
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in _OFF_VALUES
    if isinstance(value, (int, float)):
        return value > 0
    return False


def reasoning_requested(body: dict) -> bool:
    """True when the client explicitly asks for reasoning/thinking."""
    return any(_flag_on(body.get(key)) for key in ("reasoning_effort", "reasoning", "enable_thinking", "thinking"))


def required_capabilities(body: dict) -> frozenset[str]:
    """Capabilities a provider must declare to serve this request."""
    required = set(_TEXT_ONLY)
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            capability = part_capability(part)
            if capability:
                required.add(capability)
    if reasoning_requested(body):
        required.add("reasoning")
    return frozenset(required)


def media_capabilities(required: frozenset[str]) -> list[str]:
    """Capabilities beyond plain text, sorted — for logs and error messages."""
    return sorted(required - _TEXT_ONLY)
