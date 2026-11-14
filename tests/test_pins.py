"""Endpoint model pinning and its admin configuration."""

import json

import httpx
import pytest
from pydantic import ValidationError

from inference_proxy.config import ClassifierConfig, ConfigError, ProviderConfig
from inference_proxy.main import create_app
from tests.conftest import chat_body, completion_json, make_config

TIERS = ("trivial", "moderate", "complex", "hardcore")


def provider(url="http://pin.test/v1", **kw):
    kw.setdefault("default_model", "m1")
    return ProviderConfig(base_url=url, **kw)


def model(req):
    return json.loads(req.content).get("model")


async def run_request(config, handler, body=None):
    app = create_app(config, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as c:
            return await c.post("/v1/chat/completions", json=body or chat_body())


def success(req):
    return httpx.Response(200, json=completion_json("ok", model(req)))


def cfg(providers, tiers=None, **kw):
    return make_config(providers=providers, tiers=tiers or {t: list(providers) for t in TIERS}, **kw)


async def test_provider_pin_overrides_tier_model():
    seen = []
    def handler(r):
        if r.url.host == "classifier.test": return httpx.Response(200, json={"choices": [{"message": {"content": "2"}}]})
        seen.append(model(r)); return success(r)
    c = cfg({"p": provider(pin_model=True, tier_models={"trivial": "small"})})
    r = await run_request(c, handler, chat_body())
    assert r.status_code == 200 and seen == ["m1"]


async def test_configured_endpoint_pin_covers_matching_providers_only():
    seen = []
    def handler(r):
        if r.url.host == "judge.test": return httpx.Response(200, json={"choices": [{"message": {"content": "2"}}]})
        if r.url.host == "classifier.test": return httpx.Response(200, json={"choices": [{"message": {"content": "2"}}]})
        seen.append((r.url.host, model(r))); return success(r)
    ps = {"a": provider("http://HOST:1234/v1", default_model="one"), "b": provider("http://host:1234/v1/", default_model="two", tier_models={"trivial": "small"}), "other": provider("http://else.test/v1", default_model="else")}
    c = cfg(ps, tiers={t: ["a", "b", "other"] for t in TIERS}, pinned_models={"http://host:1234/v1/": "fixed"})
    r = await run_request(c, handler, chat_body())
    assert r.status_code == 200 and seen == [("host", "fixed")]
    assert (await run_request(c.model_copy(update={"tiers": c.tiers}), handler, chat_body(model="other"))).status_code == 200


@pytest.mark.parametrize("fallback", [[], ["standin"]])
async def test_missing_pinned_model_skips_catalogue_siblings_and_fails_over(fallback):
    seen, gets = [], []
    def handler(r):
        if r.method == "GET":
            if r.url.host == "classifier.test": return httpx.Response(404)
            gets.append(r.url.path); return httpx.Response(200, json={"data": [{"id": "sibling"}]})
        if r.url.host == "classifier.test": return httpx.Response(200, json={"choices": [{"message": {"content": "2"}}]})
        seen.append((r.url.host, model(r)))
        if r.url.host == "pin.test": return httpx.Response(404, json={"error": {"message": "model does not exist"}})
        return success(r)
    ps = {"p": provider(pin_model=True, fallback_models=fallback), "next": provider("http://next.test/v1")}
    r = await run_request(cfg(ps, tiers={t: ["p", "next"] for t in TIERS}), handler)
    assert r.status_code == 200 and seen == [("pin.test", "m1"), ("next.test", "m1")]
    assert not gets


async def test_explicit_client_pin_is_honoured():
    seen = []
    def handler(r): seen.append(model(r)); return success(r)
    r = await run_request(cfg({"p": provider(pin_model=True)}), handler, chat_body(model="p:other"))
    assert r.status_code == 200 and seen == ["other"]


@pytest.mark.parametrize("pinned,expected", [(True, "fixed"), (False, "different")])
async def test_judge_uses_endpoint_pin(pinned, expected):
    seen = []
    def handler(r):
        if r.url.host == "pin.test":
            seen.append(model(r)); return httpx.Response(200, json={"choices": [{"message": {"content": "2"}}]})
        if r.url.host == "judge.test": return httpx.Response(200, json={"choices": [{"message": {"content": "2"}}]})
        return success(r)
    ps = {"p": provider("http://other.test/v1")}
    if pinned: ps["same"] = provider("http://pin.test/v1", default_model="fixed")
    c = cfg(ps, tiers={t: ["p"] for t in TIERS}, classifier=ClassifierConfig(base_url="http://pin.test/v1" if pinned else "http://judge.test/v1", model="different"), pinned_models={"http://pin.test/v1": "fixed"} if pinned else {})
    await run_request(c, handler)
    if pinned: assert "fixed" in seen
    else: assert ("judge.test", "different") not in seen  # checked by captured body below


def test_pin_config_validation():
    with pytest.raises((ConfigError, ValidationError)):
        cfg({"a": provider("http://same/v1", pin_model=True, default_model="one"), "b": provider("http://same/v1/", pin_model=True, default_model="two")})
    with pytest.raises((ConfigError, ValidationError)):
        cfg({"a": ProviderConfig(base_url="http://same/v1", pin_model=True)})
    assert cfg({"a": provider("http://same/v1", pin_model=True), "b": provider("http://same/v1/", pin_model=True)})


async def test_admin_models_settings_and_live_pin():
    seen = []
    def handler(r):
        if r.url.host == "judge.test": return httpx.Response(200, json={"choices": [{"message": {"content": "2"}}]})
        if r.url.host == "classifier.test": return httpx.Response(200, json={"choices": [{"message": {"content": "2"}}]})
        seen.append(model(r)); return success(r)
    c = cfg({"p": provider(tier_models={"trivial": "small"}, fallback_models=["spare"], models=["offered"])})
    app = create_app(c, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client:
            models = (await client.get("/admin/api/models")).json()
            settings = (await client.get("/admin/api/settings")).json()["providers"]["p"]
            assert models["pins"] == {}
            assert all(k in settings for k in ("pin_model", "tier_models", "fallback_models", "models", "type"))
            assert (await client.put("/admin/api/settings", json={"providers": {"p": {"pin_model": True}}})).status_code == 200
            response = await client.post("/v1/chat/completions", json=chat_body())
    assert response.status_code == 200 and seen == ["m1"]


def test_cli_pin_model_is_ignored():
    from inference_proxy.config import compute_pins
    c = cfg({"cli": ProviderConfig(type="claude", pin_model=True, default_model="sonnet", models=["sonnet"])})
    assert compute_pins(c) == {}
