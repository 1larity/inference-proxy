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


class ProviderConfig(BaseModel):
    base_url: str = Field(min_length=1)
    api_key: str | None = None
    accounts: list[AccountConfig] = Field(default_factory=list)
    default_model: str | None = None
    type: Literal["openai"] = "openai"
    timeout_seconds: float = 120.0
    quota_cooldown_seconds: float = 600.0
    auth_failure_cooldown_seconds: float = 3600.0
    extra_body: dict[str, Any] = Field(default_factory=dict)
    max_tokens_cap: int | None = Field(default=None, ge=1)
    capabilities: list[str] = Field(default_factory=lambda: ["text"])

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


class ProxyConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    tiers: dict[str, list[str]]
    providers: dict[str, ProviderConfig]
    tier_policies: dict[str, TierPolicy] = Field(default_factory=dict)

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
        for tier in ("trivial", "moderate", "complex", "hardcore"):
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
        return self


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
