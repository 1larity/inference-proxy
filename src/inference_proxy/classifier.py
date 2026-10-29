"""Task difficulty classification via the local LM, with a heuristic fallback."""

from __future__ import annotations

import re

import httpx

from .config import ClassifierConfig
from .context import CHARS_PER_TOKEN, ContextRegistry
from .media import PART_CAPABILITIES

SYSTEM_PROMPT = (
    "You are a strict task-difficulty rater for an AI assistant. "
    "Rate how hard the user's request is on a scale of 1 to 10: "
    "1-3 = trivial (greetings, short factual questions, simple rewrites); "
    "4-6 = ordinary work (typical coding, writing, analysis, multi-step but standard); "
    "7-10 = hard (complex debugging, architecture, proofs, large multi-part tasks, "
    "specialist knowledge, long-context reasoning). "
    "Respond with a single integer from 1 to 10 and nothing else."
)

_SIMPLE_RE = re.compile(
    r"\b(hi|hello|hey|thanks|thank you|who are you|what is your name|good morning|good evening)\b",
    re.IGNORECASE,
)
_HARD_RE = re.compile(
    r"\b(prove|proof|derive|derivation|formally|optimize|optimise|architecture|algorithm"
    r"|complexity|refactor|migrate|benchmark|state[- ]of[- ]the[- ]art|root cause"
    r"|distributed|concurrency|race condition|kubernetes|compiler|cryptograph)\b",
    re.IGNORECASE,
)
_SCORE_RE = re.compile(r"\b(10|[1-9])\b")
_MEDIA_MARKER_RE = re.compile(r"\[(image|video|audio|file)\]")


def message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                capability = PART_CAPABILITIES.get(str(part.get("type", "")))
                if capability:
                    # placeholder so the judge knows media is present
                    parts.append(f"[{capability}]")
                elif part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def build_transcript(messages: list[dict]) -> str:
    lines = []
    for message in messages:
        role = str(message.get("role", "user"))
        lines.append(f"{role.capitalize()}: {message_text(message.get('content'))}")
    return "\n".join(lines)


def extract_score(text: str) -> int | None:
    match = _SCORE_RE.search(text)
    return int(match.group(1)) if match else None


def heuristic_score(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 1
    score = 2
    if len(stripped) > 400:
        score += 1
    if len(stripped) > 1500:
        score += 1
    if stripped.count("```") >= 2:
        score += 2
    score += min(len(set(_HARD_RE.findall(stripped))), 3)
    if len(set(_HARD_RE.findall(stripped))) >= 5:
        score += 1
    questions = stripped.count("?") + stripped.count("\n- ") + stripped.count("\n1. ")
    if questions >= 3:
        score += 1
    if _SIMPLE_RE.search(stripped) and len(stripped) < 120:
        score = min(score, 3)
    if _MEDIA_MARKER_RE.search(stripped):
        # media always needs at least a vision-capable model
        score = max(score, 4)
    return max(1, min(10, score))


def chat_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


def oversize_limit_chars(
    config: ClassifierConfig,
    registry: ContextRegistry | None,
) -> int | None:
    """Prompt size (chars) above which the LLM judge is skipped; None = never skip."""
    if config.context_chars is not None:
        return int(config.context_chars * config.oversize_ratio)
    if registry is None:
        return None
    tokens = registry.known_tokens(config.base_url, config.model)
    if tokens is None:
        return None
    return int(tokens * CHARS_PER_TOKEN * config.oversize_ratio)


async def classify_difficulty(
    client: httpx.AsyncClient,
    config: ClassifierConfig,
    messages: list[dict],
    context_registry: ContextRegistry | None = None,
    difficulty_floor: int = 0,
) -> tuple[int, str]:
    transcript = build_transcript(messages)
    oversize_limit = oversize_limit_chars(config, context_registry)
    if oversize_limit is not None and len(transcript) > oversize_limit:
        # Too big for the judge to see whole: judging is unreliable and risks
        # an overflow loop, so trust the length heuristic instead.
        return max(difficulty_floor, heuristic_score(transcript)), "oversize"
    transcript = transcript[: config.max_chars]
    headers = {}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    try:
        response = await client.post(
            chat_url(config.base_url),
            headers=headers,
            timeout=httpx.Timeout(config.timeout_seconds),
            json={
                "model": config.model,
                "temperature": config.temperature,
                "max_tokens": 8,
                "stream": False,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": transcript},
                ],
            },
        )
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            score = extract_score(content)
            if score is not None:
                return score, "llm"
        elif context_registry is not None:
            # the judge itself overflowed: learn its real limit for next time
            context_registry.learn_from_error(
                config.base_url, config.model, await response.aread()
            )
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
        pass
    return heuristic_score(transcript), "heuristic"
