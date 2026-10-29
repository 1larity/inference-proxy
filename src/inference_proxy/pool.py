"""Provider account pools: selection, cooldowns, and health tracking."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import ProviderConfig

QUOTA = "quota"
AUTH = "auth"
SERVER = "server"
NETWORK = "network"
REQUEST = "request"
UNSUPPORTED = "unsupported"
OVERFLOW = "overflow"


@dataclass
class Account:
    provider: str
    index: int
    api_key: str | None
    base_url: str
    note: str | None = None
    cooldown_until: float = 0.0
    consecutive_failures: int = 0
    in_flight: int = 0
    last_used: float = 0.0
    total_requests: int = 0
    total_failures: int = 0

    @property
    def id(self) -> str:
        return f"{self.provider}#{self.index}"


def _accounts_from_config(
    name: str,
    config: ProviderConfig,
    carry: dict[int, Account] | None = None,
) -> list[Account]:
    raw_accounts = config.accounts or [None]
    accounts = []
    for i, raw in enumerate(raw_accounts):
        account = Account(
            provider=name,
            index=i,
            api_key=(raw.api_key if raw else None) or config.api_key,
            base_url=(raw.base_url if raw else None) or config.base_url,
            note=raw.note if raw else None,
        )
        if carry and i in carry:
            old = carry[i]
            account.cooldown_until = old.cooldown_until
            account.consecutive_failures = old.consecutive_failures
            account.in_flight = old.in_flight
            account.last_used = old.last_used
            account.total_requests = old.total_requests
            account.total_failures = old.total_failures
        accounts.append(account)
    return accounts


class ProviderPool:
    def __init__(
        self,
        name: str,
        config: ProviderConfig,
        clock=time.time,
    ) -> None:
        self.name = name
        self.config = config
        self._clock = clock
        self.accounts: list[Account] = _accounts_from_config(name, config)

    def healthy_accounts(self) -> list[Account]:
        now = self._clock()
        return [a for a in self.accounts if a.cooldown_until <= now]

    def acquire(self) -> Account | None:
        healthy = self.healthy_accounts()
        if not healthy:
            return None
        account = min(healthy, key=lambda a: (a.in_flight, a.last_used))
        account.in_flight += 1
        account.last_used = self._clock()
        account.total_requests += 1
        return account

    def release(self, account: Account) -> None:
        account.in_flight = max(0, account.in_flight - 1)

    def report_success(self, account: Account) -> None:
        account.consecutive_failures = 0
        account.cooldown_until = 0.0
        self.release(account)

    def report_failure(self, account: Account, kind: str) -> None:
        now = self._clock()
        account.total_failures += 1
        account.consecutive_failures += 1
        if kind == QUOTA:
            cooldown = self.config.quota_cooldown_seconds
        elif kind == AUTH:
            cooldown = self.config.auth_failure_cooldown_seconds
        else:
            cooldown = min(5.0 * 2 ** (account.consecutive_failures - 1), 60.0)
        account.cooldown_until = now + cooldown
        self.release(account)

    def snapshot(self) -> dict:
        now = self._clock()
        return {
            "name": self.name,
            "accounts": [
                {
                    "id": a.id,
                    "healthy": a.cooldown_until <= now,
                    "cooldown_remaining_seconds": round(max(0.0, a.cooldown_until - now), 1),
                    "consecutive_failures": a.consecutive_failures,
                    "in_flight": a.in_flight,
                    "total_requests": a.total_requests,
                    "total_failures": a.total_failures,
                    "note": a.note,
                }
                for a in self.accounts
            ],
        }

    def rebuild_accounts(self, config: ProviderConfig) -> None:
        carry = {a.index: a for a in self.accounts}
        self.config = config
        self.accounts = _accounts_from_config(self.name, config, carry)

    def export_state(self) -> list[dict]:
        return [
            {
                "index": a.index,
                "cooldown_until": a.cooldown_until,
                "consecutive_failures": a.consecutive_failures,
                "total_requests": a.total_requests,
                "total_failures": a.total_failures,
            }
            for a in self.accounts
            if a.cooldown_until > 0 or a.consecutive_failures > 0 or a.total_requests > 0
        ]

    def apply_state(self, entries: list[dict]) -> None:
        now = self._clock()
        by_index = {a.index: a for a in self.accounts}
        for entry in entries:
            account = by_index.get(entry.get("index"))
            if account is None:
                continue
            cooldown_until = float(entry.get("cooldown_until", 0.0))
            if cooldown_until > now:
                account.cooldown_until = cooldown_until
            account.consecutive_failures = int(entry.get("consecutive_failures", 0))
            account.total_requests = int(entry.get("total_requests", 0))
            account.total_failures = int(entry.get("total_failures", 0))

    def reset_account(self, index: int) -> Account | None:
        for account in self.accounts:
            if account.index == index:
                account.cooldown_until = 0.0
                account.consecutive_failures = 0
                return account
        return None


def save_pool_state(path: str | Path, pools: dict[str, ProviderPool]) -> None:
    state = {name: pool.export_state() for name, pool in pools.items()}
    state = {name: entries for name, entries in state.items() if entries}
    Path(path).write_text(json.dumps(state))


def load_pool_state(path: str | Path, pools: dict[str, ProviderPool]) -> None:
    try:
        state = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return
    if not isinstance(state, dict):
        return
    for name, entries in state.items():
        pool = pools.get(name)
        if isinstance(entries, list) and pool is not None:
            pool.apply_state(entries)
