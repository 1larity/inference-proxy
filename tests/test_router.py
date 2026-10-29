from inference_proxy.config import ProviderConfig, RoutingConfig
from inference_proxy.pool import ProviderPool
from inference_proxy.router import TIERS, TierRouter


def make_router(**routing_overrides) -> TierRouter:
    routing = RoutingConfig(
        trivial_max=2, moderate_min=3, complex_min=6, hardcore_min=9,
        **routing_overrides,
    )
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
    tiers = {
        "trivial": ["local"],
        "moderate": ["zai"],
        "complex": ["openrouter"],
        "hardcore": ["kilo"],
    }
    return TierRouter(pools, tiers, routing)


def test_tier_for_thresholds():
    router = make_router()
    assert router.tier_for(1) == "trivial"
    assert router.tier_for(2) == "trivial"
    assert router.tier_for(3) == "moderate"
    assert router.tier_for(5) == "moderate"
    assert router.tier_for(6) == "complex"
    assert router.tier_for(8) == "complex"
    assert router.tier_for(9) == "hardcore"
    assert router.tier_for(10) == "hardcore"


def test_fallback_order_descends_then_ascends():
    router = make_router()
    assert router.fallback_order("hardcore") == ["hardcore", "complex", "moderate", "trivial"]
    assert router.fallback_order("moderate") == ["moderate", "trivial", "complex", "hardcore"]
    assert router.fallback_order("trivial") == ["trivial", "moderate", "complex", "hardcore"]


def test_fallback_order_disabled():
    router = make_router(fallback_to_other_tiers=False)
    assert router.fallback_order("hardcore") == ["hardcore"]


def test_plan_dedupes_providers():
    router = make_router()
    routing = RoutingConfig()
    tiers = {
        "trivial": ["local", "zai"],
        "moderate": ["zai"],
        "complex": ["openrouter"],
        "hardcore": ["kilo"],
    }
    router.tiers = tiers
    plan = router.plan("trivial")
    names = [pool.name for _, pool in plan]
    assert names == ["local", "zai", "openrouter", "kilo"]
    assert list(TIERS) == ["trivial", "moderate", "complex", "hardcore"]


def test_plan_text_only_keeps_everything():
    router = make_router()
    assert [p.name for _, p in router.plan("trivial", frozenset({"text"}))] == [
        "local", "zai", "openrouter", "kilo",
    ]


def test_plan_skips_incapable_providers():
    router = make_router()
    # local (trivial) and openrouter (complex) are text-only
    assert [p.name for _, p in router.plan("trivial", frozenset({"text", "image"}))] == [
        "zai", "kilo",
    ]
    # video exists only on zai, found by sliding hardcore -> complex -> moderate
    assert [p.name for _, p in router.plan("hardcore", frozenset({"text", "video"}))] == ["zai"]
    # reasoning: hardcore tier first (kilo), then zai from moderate; local and
    # openrouter drop out
    assert [p.name for _, p in router.plan("hardcore", frozenset({"text", "reasoning"}))] == [
        "kilo", "zai",
    ]


def test_plan_empty_when_capability_missing_everywhere():
    router = make_router()
    assert router.plan("moderate", frozenset({"text", "audio"})) == []
    assert router.plan("moderate", frozenset({"text", "file"})) == []


def test_plan_capability_filter_with_fallback_disabled():
    router = make_router(fallback_to_other_tiers=False)
    assert router.plan("trivial", frozenset({"text", "image"})) == []
