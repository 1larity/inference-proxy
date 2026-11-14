"""Per-provider model catalogue: what each provider offers, and which models can stand in for another."""

from __future__ import annotations

import time

import httpx

from .cli import CLI_TYPES, KNOWN_MODELS
from .pool import ProviderPool
from .upstream import build_headers

# Entries in a /models list that are not chat models.
_NOT_CHAT = ("embed", "whisper", "tts", "speech", "transcri", "moderation", "rerank", "guard", "dall-e", "image", "audio", "ocr")
MAX_SIBLINGS = 3


def is_chat_model(model_id: str) -> bool:
    low = model_id.lower()
    return not any(word in low for word in _NOT_CHAT)


def _status(entry: dict, model_id: str) -> str:
    status = (entry.get("raw", {}).get(model_id) or {}).get("status")
    return str(status.get("value") if isinstance(status, dict) else status or "")


class ModelCatalog:
    """Cached /models results. Nothing is fetched until asked, so a provider with no /models endpoint costs nothing."""

    def __init__(self) -> None:
        self.entries: dict[str, dict] = {}

    async def refresh(self, http: httpx.AsyncClient, pool: ProviderPool) -> dict:
        entry = {"models": [], "raw": {}, "fetched_at": time.time(), "error": None, "source": "configured"}
        cfg = pool.config
        if cfg.type in CLI_TYPES:
            entry["models"] = list(cfg.models or KNOWN_MODELS[cfg.type])
        else:
            account = pool.accounts[0] if pool.accounts else None
            if account is not None:
                url = account.base_url.rstrip("/") + "/models"
                try:
                    response = await http.get(url, headers=build_headers(account.api_key), timeout=10.0)
                    response.raise_for_status()
                    data = response.json().get("data", [])
                    entry["source"] = "provider /models"
                    for item in data:
                        if isinstance(item, dict) and item.get("id"):
                            entry["models"].append(str(item["id"]))
                            entry["raw"][str(item["id"])] = item
                except (httpx.HTTPError, ValueError, AttributeError) as exc:
                    entry["error"] = f"{type(exc).__name__}: {exc}"[:200]
        for configured in (cfg.default_model, *cfg.tier_models.values(), *cfg.fallback_models):
            if configured and configured not in entry["models"]:
                entry["models"].append(configured)
        self.entries[pool.name] = entry
        return entry

    async def ensure(self, http: httpx.AsyncClient, pool: ProviderPool) -> dict:
        return self.entries.get(pool.name) or await self.refresh(http, pool)

    def siblings(self, pool: ProviderPool, exclude: set[str]) -> list[str]:
        """Models of the same provider that can stand in when the chosen one is missing.

        Configured fallback_models win; otherwise chat models from the discovered list, in listed order.
        """
        configured = [m for m in pool.config.fallback_models if m not in exclude]
        if configured:
            return configured[:MAX_SIBLINGS]
        entry = self.entries.get(pool.name)
        if not entry:
            return []
        found = [m for m in entry["models"] if m not in exclude and is_chat_model(m)]
        # LM Studio and llama.cpp routers list models that are not loaded; a cold load is slow, so loaded ones go first.
        found.sort(key=lambda m: _status(entry, m) == "unloaded")
        return found[:MAX_SIBLINGS]

    def snapshot(self) -> dict:
        return {name: {k: v for k, v in e.items() if k != "raw"} for name, e in self.entries.items()}
