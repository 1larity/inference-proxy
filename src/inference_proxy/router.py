"""Tier routing: difficulty score -> tier, with exhaustion fallback ordering."""

from __future__ import annotations

from .config import RoutingConfig
from .pool import ProviderPool

TIERS = ("easy", "moderate", "hard")


class TierRouter:
    def __init__(
        self,
        pools: dict[str, ProviderPool],
        tiers: dict[str, list[str]],
        routing: RoutingConfig,
    ) -> None:
        self.pools = pools
        self.tiers = tiers
        self.routing = routing

    def tier_for(self, difficulty: int) -> str:
        if difficulty >= self.routing.hard_min:
            return "hard"
        if difficulty >= self.routing.moderate_min:
            return "moderate"
        return "easy"

    def fallback_order(self, tier: str) -> list[str]:
        order = [tier]
        if not self.routing.fallback_to_other_tiers:
            return order
        index = TIERS.index(tier)
        order.extend(reversed(TIERS[:index]))
        order.extend(TIERS[index + 1 :])
        return order

    def plan(
        self,
        tier: str,
        required: frozenset[str] = frozenset(),
    ) -> list[tuple[str, ProviderPool]]:
        """Fallback-ordered providers, skipping any missing a required capability.

        The ordering follows fallback_order(tier), so a media request starts at
        its difficulty tier and slides to neighbouring tiers to find a capable
        provider. An empty result means no configured provider can serve it.
        """
        plan = []
        seen = set()
        for tier_name in self.fallback_order(tier):
            for provider_name in self.tiers[tier_name]:
                if provider_name in seen:
                    continue
                seen.add(provider_name)
                pool = self.pools[provider_name]
                if not required <= frozenset(pool.config.capabilities):
                    continue
                plan.append((tier_name, pool))
        return plan
