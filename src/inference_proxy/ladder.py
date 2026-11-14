"""Suggest a model ladder: up to 10 rungs from the weakest and cheapest model to the strongest.

The ladder is a proposal for the four named tiers (and a table to read). Scores come from independent benchmarks,
prices from the price table, and anything pinned to one local model stays a single rung.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse


def is_local_url(url: str) -> bool:
    """True for an endpoint on this machine or the home network (localhost, private addresses, .local names)."""
    host = (urlparse(url).hostname or "").lower()
    if host == "localhost" or host.endswith(".local") or host.endswith(".lan"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback


BANDS = {"trivial": (1, 2), "moderate": (3, 5), "complex": (6, 8), "hardcore": (9, 10)}
TOLERANCE = 0.03  # a model within 3% of a rung's score is interchangeable, so the cheaper (then faster) one wins


def cost_of(c: dict) -> float:
    """Blended cost per million tokens (output weighs three times input).

    A model on this network costs nothing. A hosted model with no known price still costs a little, so a local
    model wins a tie against it.
    """
    if c.get("local"):
        return 0.0
    return 0.001 + (c.get("input_per_m") or 0.0) + 3.0 * (c.get("output_per_m") or 0.0)


def suggest(candidates: list[dict], rungs: int = 10, tolerance: float = TOLERANCE) -> dict:
    """candidates: {provider, model, score, input_per_m, output_per_m, tokens_per_s?}. Unscored ones are returned separately.

    tolerance is how much score you will give up for a cheaper or faster model (0.03 = 3%).
    """
    rungs = max(1, min(10, rungs))
    scored = sorted((c for c in candidates if c.get("score") is not None), key=lambda c: (c["score"], cost_of(c)))
    unscored = [c for c in candidates if c.get("score") is None]
    if not scored:
        return {"rungs": [], "tiers": {}, "unscored": unscored}
    chosen: list[dict] = []
    for i in range(rungs):
        q = (i + 0.5) / rungs
        pick = scored[min(len(scored) - 1, int(q * len(scored)))]
        near = [c for c in scored if pick["score"] * (1 - tolerance) <= c["score"] <= pick["score"] * (1 + tolerance)]
        # cheapest wins, then fastest (when a speed is known), then the stronger of the two
        pick = min(near, key=lambda c: (cost_of(c), -(c.get("tokens_per_s") or 0.0), -c["score"]))
        chosen.append(pick)
    ladder = []
    for i, c in enumerate(chosen):
        lo = round(1 + i * 10 / rungs)
        hi = round((i + 1) * 10 / rungs)
        ladder.append({"rung": i + 1, "difficulty_from": lo, "difficulty_to": max(lo, hi), **c})
    tiers = {}
    for tier, (lo, hi) in BANDS.items():
        mid = (lo + hi) / 2
        rung = min(ladder, key=lambda r: abs((r["difficulty_from"] + r["difficulty_to"]) / 2 - mid))
        tiers[tier] = {"provider": rung["provider"], "model": rung["model"], "score": rung["score"], "rung": rung["rung"]}
    return {"rungs": ladder, "tiers": tiers, "unscored": unscored}
