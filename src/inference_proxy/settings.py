"""Persisted UI settings: an overlay JSON deep-merged over the base YAML config.

The overlay stores edits made in the admin UI (thresholds, tier assignments,
provider accounts). API keys are never returned to the UI in plaintext: they
are masked, and a masked value sent back on save means "keep the current key".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .config import ProxyConfig, expand_env
from .pool import ProviderPool
from .router import TierRouter

MASK = "\u2022\u2022\u2022\u2022"

ROUTING_FIELDS = ("trivial_max", "moderate_min", "complex_min", "hardcore_min", "fallback_difficulty", "fallback_to_other_tiers")
CLASSIFIER_FIELDS = (
    "base_url",
    "api_key",
    "model",
    "timeout_seconds",
    "max_chars",
    "temperature",
    "context_chars",
    "oversize_ratio",
)
PROVIDER_FIELDS = (
    "base_url",
    "default_model",
    "timeout_seconds",
    "quota_cooldown_seconds",
    "auth_failure_cooldown_seconds",
    "max_tokens_cap",
    "extra_body",
    "capabilities",
)
TIER_NAMES = ("trivial", "moderate", "complex", "hardcore")
PANEL_IDS = ("routing-container", "tiers-container", "providers-container", "pools-container")


class SettingsError(Exception):
    pass


def load_overlay(path: str | Path | None) -> dict:
    if not path:
        return {}
    try:
        overlay = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return overlay if isinstance(overlay, dict) else {}


def save_overlay(path: str | Path | None, overlay: dict) -> None:
    if not path:
        return
    Path(path).write_text(json.dumps(overlay, indent=2, ensure_ascii=False))


def deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def merge_overlay_into_raw(raw: dict, overlay: dict) -> dict:
    """Deep-merge panel_states and other overlay fields into the raw config.

    Handles provider deletions first, then merges panel_states.
    """
    merged = deep_merge(raw, overlay)
    # Handle provider deletions before panel_states
    providers = merged.get("providers")
    if isinstance(providers, dict):
        deleted = {name for name, value in providers.items() if value is None}
        merged["providers"] = {
            name: value for name, value in providers.items() if value is not None
        }
        if deleted:
            tiers = merged.get("tiers")
            if isinstance(tiers, dict):
                merged["tiers"] = {
                    tier: [n for n in names if n not in deleted]
                    for tier, names in tiers.items()
                }
    # Panel states are not part of the ProxyConfig model, so handle them separately
    panel_states = overlay.get("panel_states", {})
    merged["panel_states"] = panel_states
    return merged


def mask_key(key: str | None) -> str | None:
    if not key:
        return None
    if len(key) <= 8:
        return MASK + MASK
    return key[:3] + MASK + key[-4:]


def settings_view(
    config: ProxyConfig,
    pools: dict[str, ProviderPool],
    settings_file: str | None,
) -> dict:
    result = {
        "settings_file": settings_file,
        "routing": {field: getattr(config.routing, field) for field in ROUTING_FIELDS},
        "classifier": {
            field: (
                mask_key(getattr(config.classifier, field))
                if field == "api_key"
                else getattr(config.classifier, field)
            )
            for field in CLASSIFIER_FIELDS
        },
        "server": {"api_key": mask_key(config.server.api_key)},
        "tiers": {tier: list(config.tiers.get(tier, [])) for tier in TIER_NAMES},
        "tier_policies": {
            tier: {
                "max_tokens": policy.max_tokens,
                "extra_body": policy.extra_body,
            }
            for tier, policy in config.tier_policies.items()
        },
        "providers": {
            name: {
                "base_url": provider.base_url,
                "default_model": provider.default_model,
                "timeout_seconds": provider.timeout_seconds,
                "quota_cooldown_seconds": provider.quota_cooldown_seconds,
                "auth_failure_cooldown_seconds": provider.auth_failure_cooldown_seconds,
                "max_tokens_cap": provider.max_tokens_cap,
                "extra_body": provider.extra_body,
                "capabilities": list(provider.capabilities),
                "accounts": [
                    {
                        "api_key": mask_key(account.api_key),
                        "note": account.note,
                        "has_key": bool(account.api_key),
                    }
                    for account in pools[name].accounts
                ],
            }
            for name, provider in config.providers.items()
        },
    }
    if settings_file:
        overlay = load_overlay(settings_file)
        for panel_id in PANEL_IDS:
            if panel_id in overlay and panel_id in PANEL_IDS:
                result[panel_id] = overlay[panel_id]
    return result


def build_overlay(payload: dict, pools: dict[str, ProviderPool]) -> dict:
    """Convert a UI payload into an overlay dict, resolving masked keys."""
    overlay: dict[str, Any] = {}

    routing = payload.get("routing")
    if isinstance(routing, dict):
        overlay["routing"] = {field: routing[field] for field in ROUTING_FIELDS if field in routing}

    tiers = payload.get("tiers")
    if isinstance(tiers, dict):
        overlay["tiers"] = {
            tier: list(tiers.get(tier, [])) for tier in TIER_NAMES if tier in tiers
        }

    tier_policies = payload.get("tier_policies")
    if isinstance(tier_policies, dict):
        overlay["tier_policies"] = {
            tier: policy
            for tier, policy in tier_policies.items()
            if tier in TIER_NAMES and isinstance(policy, dict)
        }

    classifier = payload.get("classifier")
    if isinstance(classifier, dict):
        overlay_classifier = {}
        for field in CLASSIFIER_FIELDS:
            if field not in classifier:
                continue
            value = classifier[field]
            if field == "api_key":
                if value and MASK in value:
                    continue  # masked -> keep current
                value = value or None
            overlay_classifier[field] = value
        if overlay_classifier:
            overlay["classifier"] = overlay_classifier

    server = payload.get("server")
    if isinstance(server, dict) and "api_key" in server:
        api_key = server["api_key"]
        if api_key and MASK in api_key:
            pass  # keep whatever is currently configured: omit from overlay
        else:
            overlay["server"] = {"api_key": api_key or None}

    providers = payload.get("providers")
    if isinstance(providers, dict):
        overlay_providers = {}
        for name, provider in providers.items():
            if provider is None:
                # null marks a provider for deletion; merge_config applies it
                overlay_providers[name] = None
                continue
            if not isinstance(provider, dict):
                continue
            entry: dict[str, Any] = {
                field: provider[field] for field in PROVIDER_FIELDS if field in provider
            }
            accounts = provider.get("accounts")
            if isinstance(accounts, list):
                existing = pools[name].accounts if name in pools else []
                resolved = []
                for i, account in enumerate(accounts):
                    key = account.get("api_key") or None
                    if key and MASK in key:
                        key = existing[i].api_key if i < len(existing) else None
                    resolved.append({"api_key": key, "note": account.get("note")})
                entry["accounts"] = resolved
            overlay_providers[name] = entry
        if overlay_providers:
            overlay["providers"] = overlay_providers

    panel_states = payload.get("panel_states")
    if isinstance(panel_states, dict):
        overlay["panel_states"] = panel_states

    return overlay


def merge_raw(raw: dict, overlay: dict) -> dict:
    """Deep-merge the overlay over the base config dict, applying deletions.

    An overlay entry `"providers": {"name": null}` deletes a provider and
    also removes it from any tier list it appears in.
    """
    merged = deep_merge(raw, expand_env(overlay))
    providers = merged.get("providers")
    if isinstance(providers, dict):
        deleted = {name for name, value in providers.items() if value is None}
        merged["providers"] = {
            name: value for name, value in providers.items() if value is not None
        }
        if deleted:
            tiers = merged.get("tiers")
            if isinstance(tiers, dict):
                merged["tiers"] = {
                    tier: [n for n in names if n not in deleted]
                    for tier, names in tiers.items()
                }
    return merged


def merge_config(raw: dict, overlay: dict) -> ProxyConfig:
    return ProxyConfig.model_validate(merge_raw(raw, overlay))


def apply_live(
    config: ProxyConfig,
    pools: dict[str, ProviderPool],
    router: TierRouter,
    new_config: ProxyConfig,
) -> None:
    """Hot-apply a validated config onto the live objects captured by routes."""
    for field in ROUTING_FIELDS:
        setattr(config.routing, field, getattr(new_config.routing, field))
    for field in CLASSIFIER_FIELDS:
        setattr(config.classifier, field, getattr(new_config.classifier, field))
    config.server.api_key = new_config.server.api_key

    config.tiers.clear()
    config.tiers.update({tier: list(names) for tier, names in new_config.tiers.items()})
    config.providers.clear()
    config.providers.update(new_config.providers)
    config.tier_policies.clear()
    config.tier_policies.update(new_config.tier_policies)

    for name, provider in new_config.providers.items():
        if name in pools:
            pools[name].rebuild_accounts(provider)
        else:
            pools[name] = ProviderPool(name, provider)
    for name in [n for n in pools if n not in new_config.providers]:
        del pools[name]

    # router shares the pools dict and config.tiers/config.routing objects,
    # so it is already up to date; this keeps its constructor args explicit.
    router.pools = pools
