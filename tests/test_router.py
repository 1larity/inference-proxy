from inference_proxy.config import ProviderConfig, RoutingConfig
from inference_proxy.pool import ProviderPool
from inference_proxy.router import TIERS, TierRouter


def make_router(**routing_overrides) -> TierRouter:
    routing = RoutingConfig(moderate_min=4, hard_min=7, **routing_overrides)
    providers = {
        "local": ProviderConfig(base_url="http://local.test/v1", capabilities=["text"]),
        "zai": ProviderConfig(
            base_url="http://zai.test/v1",
            capabilities=["text", "image", "video", "reasoning"],
        ),
        "openrouter": ProviderConfig(base_url="http://or.test/v1", capabilities=["text"]),
        "kilo": ProviderConfig(
            base_url="http://kilo.test/v1",
            capabilities=["text", "image", "reasoning"],
        ),
    }
    pools = {name: ProviderPool(name, cfg) for name, cfg in providers.items()}
    tiers = {"easy": ["local"], "moderate": ["zai"], "hard": ["openrouter", "kilo"]}
    return TierRouter(pools, tiers, routing)


def test_tier_for_thresholds():
    router = make_router()
    assert router.tier_for(1) == "easy"
    assert router.tier_for(3) == "easy"
    assert router.tier_for(4) == "moderate"
    assert router.tier_for(6) == "moderate"
    assert router.tier_for(7) == "hard"
    assert router.tier_for(10) == "hard"


def test_fallback_order_descends_then_ascends():
    router = make_router()
    assert router.fallback_order("hard") == ["hard", "moderate", "easy"]
    assert router.fallback_order("moderate") == ["moderate", "easy", "hard"]
    assert router.fallback_order("easy") == ["easy", "moderate", "hard"]


def test_fallback_order_disabled():
    router = make_router(fallback_to_other_tiers=False)
    assert router.fallback_order("hard") == ["hard"]


def test_plan_dedupes_providers():
    router = make_router()
    routing = RoutingConfig()
    tiers = {"easy": ["local", "zai"], "moderate": ["zai"], "hard": ["openrouter", "kilo"]}
    router.tiers = tiers
    plan = router.plan("easy")
    names = [pool.name for _, pool in plan]
    assert names == ["local", "zai", "openrouter", "kilo"]
    assert list(TIERS) == ["easy", "moderate", "hard"]


def test_plan_text_only_keeps_everything():
    router = make_router()
    assert [p.name for _, p in router.plan("easy", frozenset({"text"}))] == [
        "local", "zai", "openrouter", "kilo",
    ]


def test_plan_skips_incapable_providers():
    router = make_router()
    # local (easy) and openrouter (hard) are text-only
    assert [p.name for _, p in router.plan("easy", frozenset({"text", "image"}))] == [
        "zai", "kilo",
    ]
    # video exists only on zai, found by sliding hard -> moderate
    assert [p.name for _, p in router.plan("hard", frozenset({"text", "video"}))] == ["zai"]
    # reasoning: hard tier first (kilo), then zai from moderate; local and
    # openrouter drop out
    assert [p.name for _, p in router.plan("hard", frozenset({"text", "reasoning"}))] == [
        "kilo", "zai",
    ]


def test_plan_empty_when_capability_missing_everywhere():
    router = make_router()
    assert router.plan("moderate", frozenset({"text", "audio"})) == []
    assert router.plan("moderate", frozenset({"text", "file"})) == []


def test_plan_capability_filter_with_fallback_disabled():
    router = make_router(fallback_to_other_tiers=False)
    assert router.plan("easy", frozenset({"text", "image"})) == []
