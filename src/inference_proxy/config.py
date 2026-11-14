"""Configuration loading: YAML + environment variable expansion + validation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from .media import CAPABILITIES


class ConfigError(Exception):
    pass


_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(node: object) -> object:
    if isinstance(node, str):
        return _ENV_RE.sub(_substitute, node)
    if isinstance(node, list):
        return [expand_env(item) for item in node]
    if isinstance(node, dict):
        return {key: expand_env(value) for key, value in node.items()}
    return node


def _substitute(match: re.Match[str]) -> str:
    name, default = match.group(1), match.group(2)
    if name in os.environ:
        return os.environ[name]
    if default is not None:
        return default
    raise ConfigError(f"environment variable '{name}' is not set and no default was given")


class AccountConfig(BaseModel):
    api_key: str | None = None
    base_url: str | None = None
    note: str | None = None


TIER_NAMES = ("trivial", "moderate", "complex", "hardcore")


class ProviderConfig(BaseModel):
    base_url: str = ""
    api_key: str | None = None
    accounts: list[AccountConfig] = Field(default_factory=list)
    default_model: str | None = None
    # openai: any OpenAI-compatible HTTP endpoint. claude / codex: run that
    # command-line app as the provider (one login, no key rotation, no tools).
    type: Literal["openai", "claude", "codex"] = "openai"
    # Model per tier for this provider (a cheap model for trivial, a strong one
    # for hardcore). Falls back to default_model.
    tier_models: dict[str, str] = Field(default_factory=dict)
    # Tried, in order, when the chosen model is missing or retired. When empty,
    # models found through the provider's /models list are used.
    fallback_models: list[str] = Field(default_factory=list)
    # Keep this provider's endpoint on default_model only. A local server that swaps models on demand
    # (LM Studio, llama.cpp router) stalls for many seconds on each swap, so the proxy never asks it for another.
    pin_model: bool = False
    # Models offered by a provider that has no /models endpoint (the CLIs).
    models: list[str] = Field(default_factory=list)
    timeout_seconds: float = 120.0
    quota_cooldown_seconds: float = 600.0
    auth_failure_cooldown_seconds: float = 3600.0
    extra_body: dict[str, Any] = Field(default_factory=dict)
    max_tokens_cap: int | None = Field(default=None, ge=1)
    capabilities: list[str] = Field(default_factory=lambda: ["text"])

    @field_validator("tier_models")
    @classmethod
    def _check_tier_models(cls, tier_models: dict[str, str]) -> dict[str, str]:
        extra = set(tier_models) - set(TIER_NAMES)
        if extra:
            raise ConfigError(f"tier_models has unknown tiers: {sorted(extra)}")
        return tier_models

    @model_validator(mode="after")
    def _check_endpoint(self) -> ProviderConfig:
        if self.type == "openai" and not self.base_url:
            raise ConfigError("base_url is required for an openai-type provider")
        return self

    @field_validator("capabilities")
    @classmethod
    def _check_capabilities(cls, caps: list[str]) -> list[str]:
        unknown = set(caps) - set(CAPABILITIES)
        if unknown:
            raise ConfigError(
                f"unknown capabilities: {sorted(unknown)}; expected {list(CAPABILITIES)}"
            )
        if "text" not in caps:
            raise ConfigError("capabilities must include 'text'")
        return list(dict.fromkeys(caps))


class ClassifierConfig(BaseModel):
    # Which model judges difficulty. Unset: the local server below. A provider
    # name: that provider's model. "auto": the local server when it answers,
    # otherwise the first openai-type provider.
    provider: str | None = None
    base_url: str = Field(default="http://127.0.0.1:1234/v1", min_length=1)
    api_key: str | None = None
    model: str = "local"
    timeout_seconds: float = 15.0
    max_chars: int = 6000
    temperature: float = 0.0
    # Manual override for the judge's context window, in characters. When
    # unset, the size is probed from LM Studio and learned from overflow
    # errors at runtime.
    context_chars: int | None = Field(default=None, ge=1)
    # Prompts longer than oversize_ratio of the judge context skip the LLM
    # judge and route via the length heuristic instead.
    oversize_ratio: float = Field(default=0.5, gt=0.0, le=1.0)


class TierPolicy(BaseModel):
    """Per-tier request defaults. Client-supplied values always win."""

    max_tokens: int | None = Field(default=None, ge=1)
    extra_body: dict[str, Any] = Field(default_factory=dict)


class RoutingConfig(BaseModel):
    trivial_max: int = Field(default=2, ge=1, le=10)
    moderate_min: int = Field(default=3, ge=1, le=10)
    complex_min: int = Field(default=6, ge=1, le=10)
    hardcore_min: int = Field(default=9, ge=1, le=10)
    fallback_difficulty: int = Field(default=5, ge=1, le=10)
    fallback_to_other_tiers: bool = True

    @model_validator(mode="after")
    def _check_thresholds(self) -> RoutingConfig:
        if not (
            self.trivial_max < self.moderate_min
            < self.complex_min
            < self.hardcore_min
        ):
            raise ConfigError(
                "routing thresholds must satisfy "
                "trivial_max < moderate_min < complex_min < hardcore_min"
            )
        return self


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8480
    api_key: str | None = None
    state_file: str | None = None
    settings_file: str | None = "ui-settings.json"
    # SQLite file for the price table and the usage ledger. Unset: kept in memory only.
    data_db: str | None = None
    # A price older than this many days is shown as stale.
    price_stale_days: int = Field(default=14, ge=1)


class BenchmarksConfig(BaseModel):
    # Your own free Artificial Analysis API key (https://artificialanalysis.ai). Credit is shown in the admin page.
    aa_api_key: str | None = None


class ResearchConfig(BaseModel):
    """Off by default. When on, the proxy refreshes scores and prices on a schedule and may ask a Claude CLI provider to
    read pricing pages. It only ever creates proposals, which you apply or reject in the admin page."""

    enabled: bool = False
    cli_provider: str | None = None   # a provider with type: claude
    every_days: int = Field(default=7, ge=1)
    price_pages: dict[str, str] = Field(default_factory=dict)  # provider name -> pricing page URL


def fill_missing_tiers(tiers: dict[str, list[str]], providers: list[str]) -> dict[str, list[str]]:
    """Tiers left out of the config take the providers of the nearest tier that lists some.

    A tier that is present but empty stays empty (that is an error, not a gap).
    With no tiers listed at all, every tier uses every provider in declared order.
    """
    if not tiers:
        return {t: list(providers) for t in TIER_NAMES}
    out = {t: list(names) for t, names in tiers.items()}
    for i, tier in enumerate(TIER_NAMES):
        if tier in tiers:
            continue
        for dist in range(1, len(TIER_NAMES)):
            near = [TIER_NAMES[j] for j in (i + dist, i - dist) if 0 <= j < len(TIER_NAMES) and tiers.get(TIER_NAMES[j])]
            if near:
                out[tier] = list(tiers[near[0]])
                break
    return out


class ProxyConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    tiers: dict[str, list[str]] = Field(default_factory=dict)
    providers: dict[str, ProviderConfig]
    tier_policies: dict[str, TierPolicy] = Field(default_factory=dict)
    benchmarks: BenchmarksConfig = Field(default_factory=lambda: BenchmarksConfig())
    research: ResearchConfig = Field(default_factory=lambda: ResearchConfig())
    # Endpoint base URL -> the one model the proxy may ask it for (also pins the difficulty judge on that endpoint).
    pinned_models: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _default_tiers(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("providers"):
            tiers = data.get("tiers") or {}
            if isinstance(tiers, dict) and set(tiers) <= set(TIER_NAMES) and not all(t in tiers for t in TIER_NAMES):
                data = dict(data)
                data["tiers"] = fill_missing_tiers(tiers, list(data["providers"]))
        return data

    @field_validator("tier_policies")
    @classmethod
    def _check_tier_policies(cls, policies: dict[str, TierPolicy]) -> dict[str, TierPolicy]:
        extra = set(policies) - {"trivial", "moderate", "complex", "hardcore"}
        if extra:
            raise ConfigError(f"tier_policies has unknown tiers: {sorted(extra)}")
        return policies

    @field_validator("tiers")
    @classmethod
    def _check_tiers(cls, tiers: dict[str, list[str]]) -> dict[str, list[str]]:
        for tier in TIER_NAMES:
            if tier not in tiers or not tiers[tier]:
                raise ConfigError(f"tiers.{tier} must list at least one provider")
        extra = set(tiers) - {"trivial", "moderate", "complex", "hardcore"}
        if extra:
            raise ConfigError(f"unknown tiers: {sorted(extra)}; expected trivial/moderate/complex/hardcore")
        return tiers

    @model_validator(mode="after")
    def _check_providers(self) -> ProxyConfig:
        unknown = {
            provider
            for names in self.tiers.values()
            for provider in names
            if provider not in self.providers
        }
        if unknown:
            raise ConfigError(f"tiers reference unknown providers: {sorted(unknown)}")
        compute_pins(self)  # a pin conflict should stop startup, not surface on the first request
        return self


def endpoint_key(url: str) -> str:
    return url.strip().rstrip("/").lower()


def compute_pins(config: ProxyConfig) -> dict[str, str]:
    """Endpoint -> pinned model, from pinned_models and providers marked pin_model."""
    pins = {endpoint_key(url): model for url, model in config.pinned_models.items() if model}
    for name, provider in config.providers.items():
        if not provider.pin_model or provider.type != "openai":
            continue
        if not provider.default_model:
            raise ConfigError(f"provider '{name}' sets pin_model but has no default_model")
        key = endpoint_key(provider.base_url)
        if pins.get(key, provider.default_model) != provider.default_model:
            raise ConfigError(f"endpoint {provider.base_url} is pinned to two models: {pins[key]} and {provider.default_model}")
        pins[key] = provider.default_model
    return pins


def load_config_dict(path: str | Path) -> dict:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from None
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping, got {type(raw).__name__}")
    return expand_env(raw)


def load_config(path: str | Path) -> ProxyConfig:
    return ProxyConfig.model_validate(load_config_dict(path))
