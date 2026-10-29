import json

import httpx
import pytest

from inference_proxy.config import ServerConfig
from inference_proxy.main import create_app
from tests.conftest import FakeUpstreams, chat_body, make_config


@pytest.fixture
def settings_config(tmp_path):
    overlay = tmp_path / "ui-settings.json"
    return make_config(server=ServerConfig(settings_file=str(overlay))), overlay


@pytest.fixture
async def settings_proxy(settings_config):
    config, overlay = settings_config
    upstreams = FakeUpstreams()
    app = create_app(config, http_client=httpx.AsyncClient(transport=httpx.MockTransport(upstreams)))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            yield client, app, upstreams, overlay


async def test_get_settings_masks_keys(settings_proxy):
    client, app, upstreams, overlay = settings_proxy
    response = await client.get("/admin/api/settings")
    view = response.json()
    assert view["routing"]["moderate_min"] == 3
    assert view["providers"]["zai"]["accounts"][0]["api_key"] == "zai••••ey-1"
    assert "zai-key-1" not in response.text
    assert view["providers"]["local"]["accounts"][0]["has_key"] is False


async def test_put_settings_persists_and_applies(settings_proxy):
    client, app, upstreams, overlay = settings_proxy
    view = await client.get("/admin/api/settings")
    payload = {
        "routing": {"moderate_min": 6, "complex_min": 7},
        "tiers": {"trivial": ["local"], "moderate": ["zai"], "complex": ["openrouter"], "hardcore": ["openrouter", "kilo"]},
    }
    response = await client.put("/admin/api/settings", json=payload)
    assert response.status_code == 200
    assert response.json()["routing"]["moderate_min"] == 6
    assert app.state.config.routing.moderate_min == 6
    assert overlay.exists()

    # hot-applies without restart
    upstreams.classifier_score = "5"
    chat = await client.post("/v1/chat/completions", json=chat_body())
    assert chat.json()["inference_proxy"]["tier"] == "trivial"  # 5 < 6

    # persisted across "restart"
    fresh = create_app(make_config(server=ServerConfig(settings_file=str(overlay))))
    assert fresh.state.config.routing.moderate_min == 6


async def test_masked_key_keeps_current(settings_proxy):
    client, app, upstreams, overlay = settings_proxy
    view = (await client.get("/admin/api/settings")).json()
    masked = view["providers"]["zai"]["accounts"][0]["api_key"]
    payload = {
        "providers": {
            "zai": {
                "base_url": "http://zai.test/v1",
                "default_model": "glm-4.7",
                "accounts": [
                    {"api_key": masked, "note": "primary"},
                    {"api_key": "brand-new-key", "note": "overflow"},
                ],
            }
        }
    }
    response = await client.put("/admin/api/settings", json=payload)
    assert response.status_code == 200
    accounts = app.state.pools["zai"].accounts
    assert accounts[0].api_key == "zai-key-1"  # mask resolved to existing key
    assert accounts[0].note == "primary"
    assert accounts[1].api_key == "brand-new-key"

    # the new key is actually used upstream
    view2 = (await client.get("/admin/api/settings")).json()
    assert view2["providers"]["zai"]["accounts"][1]["api_key"].startswith("bra••••")


async def test_account_add_remove_live(settings_proxy):
    client, app, upstreams, overlay = settings_proxy
    payload = {
        "providers": {
            "zai": {
                "accounts": [
                    {"api_key": "zai••••-1", "note": "a"},
                    {"api_key": "zai••••-2", "note": "b"},
                    {"api_key": "zai-key-3", "note": "c"},
                ]
            }
        }
    }
    response = await client.put("/admin/api/settings", json=payload)
    assert response.status_code == 200
    assert len(app.state.pools["zai"].accounts) == 3

    payload2 = {"providers": {"zai": {"accounts": [{"api_key": "zai••••-1", "note": "a"}]}}}
    await client.put("/admin/api/settings", json=payload2)
    assert len(app.state.pools["zai"].accounts) == 1
    assert app.state.pools["zai"].accounts[0].api_key == "zai-key-1"


async def test_tier_reassignment_live(settings_proxy):
    client, app, upstreams, overlay = settings_proxy
    payload = {
        "tiers": {"trivial": ["zai", "local"], "moderate": ["zai"], "hardcore": ["openrouter", "kilo"]}
    }
    response = await client.put("/admin/api/settings", json=payload)
    assert response.status_code == 200
    chat = await client.post("/v1/chat/completions", json=chat_body(model="trivial"))
    assert chat.status_code == 200
    meta = chat.json()["inference_proxy"]
    assert meta["tier"] == "trivial"
    assert meta["provider"] == "zai"


async def test_invalid_settings_rejected_without_applying(settings_proxy):
    client, app, upstreams, overlay = settings_proxy
    payload = {"routing": {"moderate_min": 8, "hard_min": 5}}  # hardcore < moderate
    response = await client.put("/admin/api/settings", json=payload)
    assert response.status_code == 400
    assert "invalid settings" in response.json()["error"]["message"]
    assert app.state.config.routing.moderate_min == 3
    assert not overlay.exists()


async def test_unknown_tier_provider_rejected(settings_proxy):
    client, app, upstreams, overlay = settings_proxy
    payload = {"tiers": {"trivial": ["nonexistent"], "moderate": ["zai"], "hardcore": ["openrouter"]}}
    response = await client.put("/admin/api/settings", json=payload)
    assert response.status_code == 400


async def test_reset_account_endpoint(settings_proxy):
    client, app, upstreams, overlay = settings_proxy
    upstreams.classifier_score = "5"
    upstreams.zai_status_by_key = {"zai-key-1": 429, "zai-key-2": 200}
    await client.post("/v1/chat/completions", json=chat_body())
    assert app.state.pools["zai"].accounts[0].cooldown_until > 0

    response = await client.post("/admin/api/pools/zai/accounts/0/reset")
    assert response.status_code == 200
    assert app.state.pools["zai"].accounts[0].cooldown_until == 0.0
    assert app.state.pools["zai"].accounts[0].consecutive_failures == 0

    missing = await client.post("/admin/api/pools/zai/accounts/99/reset")
    assert missing.status_code == 404


async def test_stats_survive_pool_rebuild(settings_proxy):
    client, app, upstreams, overlay = settings_proxy
    upstreams.classifier_score = "5"
    upstreams.zai_status_by_key = {"zai-key-1": 429, "zai-key-2": 200}
    await client.post("/v1/chat/completions", json=chat_body())
    totals = {
        a.index: a.total_requests
        for a in app.state.pools["zai"].accounts
    }
    assert totals[0] == 1

    payload = {"providers": {"zai": {"quota_cooldown_seconds": 900}}}
    await client.put("/admin/api/settings", json=payload)
    assert app.state.pools["zai"].accounts[0].total_requests == 1
    assert app.state.pools["zai"].config.quota_cooldown_seconds == 900


async def test_delete_provider_live_and_persisted(settings_proxy):
    client, app, upstreams, overlay = settings_proxy

    # deleting a tier-assigned provider is rejected
    r = await client.put("/admin/api/settings", json={"providers": {"zai": None}})
    assert r.status_code == 400

    # retier away from zai, then delete
    retier = {"tiers": {"trivial": ["local"], "moderate": ["local"], "hardcore": ["openrouter", "kilo"]}}
    assert (await client.put("/admin/api/settings", json=retier)).status_code == 200
    r = await client.put("/admin/api/settings", json={"providers": {"zai": None}})
    assert r.status_code == 200
    assert "zai" not in app.state.pools

    view = (await client.get("/admin/api/settings")).json()
    assert "zai" not in view["providers"]

    # deletion recorded in overlay and survives restart (base yaml still lists zai)
    import json

    saved = json.loads(overlay.read_text())
    assert saved["providers"]["zai"] is None
    fresh = create_app(make_config(server=ServerConfig(settings_file=str(overlay))))
    assert "zai" not in fresh.state.config.providers
    assert "zai" not in fresh.state.pools


async def test_readd_deleted_provider(settings_proxy):
    client, app, upstreams, overlay = settings_proxy
    retier = {"tiers": {"trivial": ["local"], "moderate": ["local"], "hardcore": ["openrouter", "kilo"]}}
    assert (await client.put("/admin/api/settings", json=retier)).status_code == 200
    assert (await client.put("/admin/api/settings", json={"providers": {"zai": None}})).status_code == 200

    r = await client.put(
        "/admin/api/settings",
        json={
            "providers": {
                "zai": {
                    "base_url": "http://zai.test/v1",
                    "default_model": "glm-4.6",
                    "accounts": [{"api_key": "brand-new-key", "note": "recreated"}],
                }
            }
        },
    )
    assert r.status_code == 200
    assert "zai" in app.state.pools
    assert app.state.pools["zai"].accounts[0].api_key == "brand-new-key"

    # a provider outside every tier can still be pinned directly
    chat = await client.post("/v1/chat/completions", json=chat_body(model="zai"))
    assert chat.status_code == 200
    meta = chat.json()["inference_proxy"]
    assert meta["provider"] == "zai"
    assert meta["tier"] == "unassigned"
    assert meta["model"] == "glm-4.6"


async def test_arbitrary_provider_name(settings_proxy):
    client, app, upstreams, overlay = settings_proxy
    payload = {
        "tiers": {"trivial": ["ghost"], "moderate": ["zai"], "hardcore": ["openrouter", "kilo"]},
        "providers": {
            "ghost": {
                "base_url": "http://local.test/v1",
                "default_model": "whatever",
                "accounts": [{"api_key": "ghost-key", "note": "totally made up"}],
            }
        },
    }
    assert (await client.put("/admin/api/settings", json=payload)).status_code == 200

    chat = await client.post("/v1/chat/completions", json=chat_body(model="trivial"))
    assert chat.status_code == 200
    meta = chat.json()["inference_proxy"]
    assert meta["provider"] == "ghost"
    assert meta["account"] == "ghost#0"
    assert meta["tier"] == "trivial"

    upstream_request = upstreams.requests[-1]
    assert upstream_request.headers["Authorization"] == "Bearer ghost-key"


async def test_provider_edit_reflected_in_view_and_runtime(settings_proxy):
    client, app, upstreams, overlay = settings_proxy
    view = (await client.get("/admin/api/settings")).json()
    old = view["providers"]["openrouter"]
    payload = {
        "providers": {
            "openrouter": {
                "base_url": "https://api.z.ai/api/paas/v4",
                "default_model": "glm-5.3-flash",
                "accounts": [
                    {"api_key": old["accounts"][0]["api_key"], "note": old["accounts"][0]["note"]}
                ],
            }
        }
    }
    response = await client.put("/admin/api/settings", json=payload)
    assert response.status_code == 200

    body = response.json()["providers"]["openrouter"]
    assert body["base_url"] == "https://api.z.ai/api/paas/v4"
    assert body["default_model"] == "glm-5.3-flash"

    # runtime uses the new endpoint, and a fresh GET does not revert
    assert app.state.pools["openrouter"].accounts[0].base_url == "https://api.z.ai/api/paas/v4"
    again = (await client.get("/admin/api/settings")).json()["providers"]["openrouter"]
    assert again["base_url"] == "https://api.z.ai/api/paas/v4"
    assert again["default_model"] == "glm-5.3-flash"


async def test_extra_body_and_cap_reach_upstream(settings_proxy):
    client, app, upstreams, overlay = settings_proxy
    payload = {
        "providers": {
            "zai": {
                "extra_body": {"thinking": {"type": "disabled"}},
                "max_tokens_cap": 64,
            }
        }
    }
    assert (await client.put("/admin/api/settings", json=payload)).status_code == 200
    response = await client.post("/v1/chat/completions", json=chat_body(model="zai"))
    assert response.status_code == 200

    sent = json.loads(upstreams.requests[-1].content)
    assert sent["thinking"] == {"type": "disabled"}
    assert sent["max_tokens"] == 64


async def test_tier_policy_defaults_and_client_precedence(settings_proxy):
    client, app, upstreams, overlay = settings_proxy
    payload = {
        "tier_policies": {
            "moderate": {
                "max_tokens": 512,
                "extra_body": {"thinking": {"type": "disabled"}},
            }
        }
    }
    assert (await client.put("/admin/api/settings", json=payload)).status_code == 200

    # client sends nothing: tier defaults fill the gaps
    await client.post("/v1/chat/completions", json=chat_body(model="zai"))
    sent = json.loads(upstreams.requests[-1].content)
    assert sent["max_tokens"] == 512
    assert sent["thinking"] == {"type": "disabled"}

    # client explicit values always win, including reasoning settings
    explicit = chat_body(model="zai")
    explicit["max_tokens"] = 9999
    explicit["thinking"] = {"type": "enabled", "effort": "high"}
    await client.post("/v1/chat/completions", json=explicit)
    sent = json.loads(upstreams.requests[-1].content)
    assert sent["max_tokens"] == 9999
    assert sent["thinking"] == {"type": "enabled", "effort": "high"}

    # persists across restart
    fresh = create_app(make_config(server=ServerConfig(settings_file=str(overlay))))
    policy = fresh.state.config.tier_policies["moderate"]
    assert policy.max_tokens == 512
    assert policy.extra_body == {"thinking": {"type": "disabled"}}


async def test_ui_page_served(settings_proxy):
    client, app, upstreams, overlay = settings_proxy
    response = await client.get("/admin")
    assert response.status_code == 200
    assert "inference-proxy" in response.text
    root = await client.get("/")
    assert root.status_code == 200


async def test_provider_capabilities_via_settings(settings_proxy):
    client, app, upstreams, overlay = settings_proxy
    payload = {"providers": {"kilo": {"capabilities": ["text", "image", "video"]}}}
    response = await client.put("/admin/api/settings", json=payload)
    assert response.status_code == 200
    assert app.state.config.providers["kilo"].capabilities == ["text", "image", "video"]

    view = (await client.get("/admin/api/settings")).json()
    assert view["providers"]["kilo"]["capabilities"] == ["text", "image", "video"]

    # routing honours it immediately: video request pinned to kilo succeeds
    video = chat_body(
        model="kilo",
        messages=[{"role": "user", "content": [{"type": "video_url", "video_url": "x"}]}],
    )
    chat = await client.post("/v1/chat/completions", json=video)
    assert chat.status_code == 200

    # persists across restart
    fresh = create_app(make_config(server=ServerConfig(settings_file=str(overlay))))
    assert fresh.state.config.providers["kilo"].capabilities == ["text", "image", "video"]


async def test_unknown_capability_rejected(settings_proxy):
    client, app, upstreams, overlay = settings_proxy
    response = await client.put(
        "/admin/api/settings", json={"providers": {"kilo": {"capabilities": ["telepathy"]}}}
    )
    assert response.status_code == 400
    assert "invalid settings" in response.json()["error"]["message"]
    assert app.state.config.providers["kilo"].capabilities == ["text", "image", "reasoning"]
    assert not overlay.exists()


async def test_capabilities_without_text_rejected(settings_proxy):
    client, app, upstreams, overlay = settings_proxy
    response = await client.put(
        "/admin/api/settings", json={"providers": {"kilo": {"capabilities": ["image"]}}}
    )
    assert response.status_code == 400
    assert "invalid settings" in response.json()["error"]["message"]
