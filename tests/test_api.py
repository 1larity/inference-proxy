import httpx
import pytest

from inference_proxy.config import ClassifierConfig, RoutingConfig, ServerConfig
from inference_proxy.main import create_app
from tests.conftest import FakeUpstreams, chat_body, make_config


@pytest.fixture
async def auth_proxy():
    config = make_config(server=ServerConfig(api_key="proxy-secret"))
    upstreams = FakeUpstreams()
    app = create_app(config, http_client=httpx.AsyncClient(transport=httpx.MockTransport(upstreams)))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            yield client, upstreams


async def test_healthz(proxy):
    client, _, _ = proxy
    response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


async def test_models_lists_targets(proxy):
    client, _, _ = proxy
    response = await client.get("/v1/models")
    ids = {entry["id"] for entry in response.json()["data"]}
    assert {"auto", "trivial", "moderate", "hardcore", "local", "zai", "openrouter", "kilo"} <= ids
    assert "zai:glm-4.7" in ids


async def test_auto_easy_routes_to_local(proxy):
    client, app, upstreams = proxy
    upstreams.classifier_score = "2"
    response = await client.post("/v1/chat/completions", json=chat_body())
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "local-reply"
    assert data["inference_proxy"]["tier"] == "trivial"
    assert data["inference_proxy"]["difficulty"] == 2
    assert data["inference_proxy"]["difficulty_method"] == "llm"
    assert data["inference_proxy"]["provider"] == "local"
    assert response.headers["X-Proxy-Tier"] == "trivial"


async def test_auto_moderate_rotates_account_on_quota_error(proxy):
    client, app, upstreams = proxy
    upstreams.classifier_score = "5"
    upstreams.zai_status_by_key = {"zai-key-1": 429, "zai-key-2": 200}
    response = await client.post("/v1/chat/completions", json=chat_body())
    assert response.status_code == 200
    meta = response.json()["inference_proxy"]
    assert meta["tier"] == "moderate"
    assert meta["provider"] == "zai"
    assert meta["account"] == "zai#1"

    pools = app.state.pools
    assert pools["zai"].accounts[0].cooldown_until > 0
    assert pools["zai"].accounts[1].cooldown_until == 0


async def test_hard_tier_skips_classifier(proxy):
    client, _, upstreams = proxy
    response = await client.post("/v1/chat/completions", json=chat_body(model="hardcore"))
    assert response.status_code == 200
    assert upstreams.classifier_calls == 0
    meta = response.json()["inference_proxy"]
    assert meta["tier"] == "hardcore"
    assert meta["provider"] == "openrouter"
    assert meta["difficulty"] is None
    assert meta["difficulty_method"] == "forced"


async def test_pinned_provider_and_model(proxy):
    client, _, upstreams = proxy
    response = await client.post(
        "/v1/chat/completions", json=chat_body(model="kilo:custom-model")
    )
    assert response.status_code == 200
    meta = response.json()["inference_proxy"]
    assert meta["provider"] == "kilo"
    assert meta["model"] == "custom-model"
    upstream_request = upstreams.requests[-1]
    assert upstream_request.headers["Authorization"] == "Bearer kilo-key-1"


async def test_unknown_model_returns_400(proxy):
    client, _, _ = proxy
    response = await client.post("/v1/chat/completions", json=chat_body(model="gpt-99"))
    assert response.status_code == 400


async def test_missing_messages_returns_400(proxy):
    client, _, _ = proxy
    response = await client.post("/v1/chat/completions", json={"model": "auto"})
    assert response.status_code == 400


async def test_pin_failure_returns_502_with_attempts(proxy):
    client, _, upstreams = proxy
    upstreams.kilo_status = 503
    response = await client.post("/v1/chat/completions", json=chat_body(model="kilo"))
    assert response.status_code == 502
    error = response.json()["error"]
    assert error["attempts"][0]["status"] == 503
    assert error["attempts"][0]["kind"] == "server"


async def test_tier_fallback_when_all_moderate_exhausted(proxy):
    client, _, upstreams = proxy
    upstreams.classifier_score = "5"
    upstreams.zai_status_by_key = {"zai-key-1": 402, "zai-key-2": 402}
    response = await client.post("/v1/chat/completions", json=chat_body())
    assert response.status_code == 200
    meta = response.json()["inference_proxy"]
    assert meta["tier"] == "trivial"
    assert meta["provider"] == "local"


async def test_client_error_relays_upstream_status(proxy):
    client, _, upstreams = proxy
    upstreams.classifier_score = "1"
    upstreams.local_status = 400
    response = await client.post("/v1/chat/completions", json=chat_body())
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["provider"] == "local"


async def test_streaming_passthrough(proxy):
    client, _, upstreams = proxy

    def sse_handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "local.test":
            return upstreams.sse_response()
        return httpx.Response(404)

    upstreams.local_status = 200
    app = create_app(
        make_config(),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(sse_handler)),
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions", json=chat_body(model="trivial", stream=True)
            )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["X-Proxy-Tier"] == "trivial"
    assert b'"inference_proxy"' in response.content  # routing comment for debugging
    assert b"data: [DONE]" in response.content


async def test_proxy_auth_enforced(auth_proxy):
    client, _ = auth_proxy
    denied = await client.post("/v1/chat/completions", json=chat_body(model="trivial"))
    assert denied.status_code == 401

    allowed = await client.post(
        "/v1/chat/completions",
        json=chat_body(model="trivial"),
        headers={"Authorization": "Bearer proxy-secret"},
    )
    assert allowed.status_code == 200


async def test_recent_requests_endpoint(proxy):
    client, _, upstreams = proxy
    upstreams.classifier_score = "2"
    await client.post("/v1/chat/completions", json=chat_body(model="auto"))
    response = await client.get("/admin/api/recent")
    entries = response.json()["requests"]
    assert entries, "expected at least one recorded request"
    latest = entries[0]
    assert latest["model"] == "auto"
    assert latest["provider"] == "local"
    assert latest["tier"] == "trivial"
    assert latest["difficulty"] == 2
    assert latest["status"] == 200
    assert "upstream_model" in latest and latest["ms"] >= 0


async def test_admin_pools_endpoint(proxy):
    client, app, upstreams = proxy
    upstreams.classifier_score = "5"
    await client.post("/v1/chat/completions", json=chat_body())
    response = await client.get("/admin/pools")
    providers = {entry["name"]: entry for entry in response.json()["providers"]}
    assert providers["zai"]["accounts"][0]["total_requests"] == 1
    assert providers["zai"]["accounts"][0]["total_failures"] == 1
    assert providers["zai"]["accounts"][1]["total_failures"] == 0


def media_body(*parts: dict, **overrides) -> dict:
    return chat_body(messages=[{"role": "user", "content": list(parts)}], **overrides)


def image_part() -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}


async def test_image_request_routes_to_capable_provider(proxy):
    client, _, upstreams = proxy
    upstreams.classifier_score = "1"  # trivial tier, but local cannot take images
    response = await client.post(
        "/v1/chat/completions",
        json=media_body({"type": "text", "text": "what is this"}, image_part()),
    )
    assert response.status_code == 200
    meta = response.json()["inference_proxy"]
    assert meta["provider"] == "zai"  # nearest image-capable across tiers
    assert meta["tier"] == "moderate"
    assert meta["capabilities"] == ["image"]
    assert response.headers["X-Proxy-Provider"] == "zai"


async def test_video_request_slides_to_video_provider(proxy):
    client, _, upstreams = proxy
    upstreams.classifier_score = "9"  # hardcore tier has no video provider
    response = await client.post(
        "/v1/chat/completions",
        json=media_body({"type": "video_url", "video_url": "https://v.test/x.mp4"}),
    )
    assert response.status_code == 200
    meta = response.json()["inference_proxy"]
    assert meta["provider"] == "zai"  # only video-capable provider, found via fallback
    assert meta["difficulty"] == 9


async def test_media_without_capable_provider_rejected(proxy):
    client, _, upstreams = proxy
    response = await client.post(
        "/v1/chat/completions",
        json=media_body({"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}),
    )
    assert response.status_code == 400
    assert "audio" in response.json()["error"]["message"]
    chat_hosts = {"local.test", "zai.test", "or.test", "kilo.test"}
    assert all(r.url.host not in chat_hosts for r in upstreams.requests)


async def test_forced_tier_slides_to_capable_tier(proxy):
    client, _, upstreams = proxy
    response = await client.post(
        "/v1/chat/completions", json=media_body(image_part(), model="trivial")
    )
    assert response.status_code == 200
    meta = response.json()["inference_proxy"]
    assert meta["provider"] == "zai"  # trivial is text-only; fallback finds zai


async def test_forced_tier_without_fallback_rejects_incapable(proxy):
    config = make_config(
        routing=RoutingConfig(trivial_max=2, moderate_min=3, complex_min=6, hardcore_min=9, fallback_to_other_tiers=False)
    )
    upstreams = FakeUpstreams()
    app = create_app(config, http_client=httpx.AsyncClient(transport=httpx.MockTransport(upstreams)))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions", json=media_body(image_part(), model="trivial")
            )
    assert response.status_code == 400
    assert "tier 'trivial'" in response.json()["error"]["message"]
    assert all(r.url.host != "local.test" for r in upstreams.requests)


async def test_reasoning_request_skips_non_reasoning_providers(proxy):
    client, _, upstreams = proxy
    upstreams.classifier_score = "1"
    body = chat_body()
    body["reasoning_effort"] = "high"
    response = await client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    meta = response.json()["inference_proxy"]
    assert meta["provider"] == "zai"  # local (trivial) has no reasoning capability
    assert meta["capabilities"] == ["reasoning"]


async def test_reasoning_disabled_body_needs_no_capability(proxy):
    client, _, upstreams = proxy
    upstreams.classifier_score = "1"
    body = chat_body()
    body["thinking"] = {"type": "disabled"}
    response = await client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    assert response.json()["inference_proxy"]["provider"] == "local"


async def test_upstream_media_rejection_fails_over(proxy):
    client, app, upstreams = proxy
    upstreams.classifier_score = "5"
    upstreams.media_reject_hosts = {"zai.test"}
    response = await client.post("/v1/chat/completions", json=media_body(image_part()))
    assert response.status_code == 200
    meta = response.json()["inference_proxy"]
    assert meta["provider"] == "kilo"  # zai rejected the image despite declaring it
    assert meta["tier"] == "hardcore"
    assert [a.cooldown_until > 0 for a in app.state.pools["zai"].accounts] == [True, True]


async def test_pinned_provider_relays_media_rejection(proxy):
    client, _, upstreams = proxy
    upstreams.media_reject_hosts = {"local.test"}
    response = await client.post(
        "/v1/chat/completions", json=media_body(image_part(), model="local")
    )
    assert response.status_code == 400  # pinned means pinned: relayed, not failed over
    assert response.json()["error"]["provider"] == "local"


async def test_recent_entry_carries_capabilities(proxy):
    client, _, upstreams = proxy
    upstreams.classifier_score = "1"
    await client.post(
        "/v1/chat/completions",
        json=media_body({"type": "text", "text": "look"}, image_part()),
    )
    latest = (await client.get("/admin/api/recent")).json()["requests"][0]
    assert latest["capabilities"] == ["image"]


async def test_oversize_prompt_skips_judge_and_routes_moderate():
    config = make_config(
        classifier=ClassifierConfig(
            base_url="http://classifier.test/v1", model="rater", context_chars=1000
        )
    )
    upstreams = FakeUpstreams()
    app = create_app(config, http_client=httpx.AsyncClient(transport=httpx.MockTransport(upstreams)))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            body = chat_body(messages=[{"role": "user", "content": "x" * 2000}])
            response = await client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    meta = response.json()["inference_proxy"]
    assert meta["difficulty_method"] == "oversize"
    assert meta["tier"] == "moderate"
    assert meta["provider"] == "zai"
    assert upstreams.classifier_calls == 0


async def test_context_overflow_fails_over_without_cooldown(proxy):
    client, app, upstreams = proxy
    upstreams.classifier_score = "2"  # trivial tier -> local
    upstreams.overflow_hosts = {"local.test"}
    response = await client.post("/v1/chat/completions", json=chat_body())
    assert response.status_code == 200
    meta = response.json()["inference_proxy"]
    assert meta["provider"] == "zai"  # failed over to the moderate tier
    local = app.state.pools["local"].accounts[0]
    assert local.cooldown_until == 0  # the account is healthy; no cooldown poisoning
    assert local.consecutive_failures == 0
    registry = app.state.context_registry
    assert registry.known_tokens("http://local.test/v1", "qwen3.5") == 8192
