"""Provider model selection, fallback, catalogues, and CLI integration."""

import json
import os

import httpx
import pytest
from pydantic import ValidationError

from inference_proxy.config import (
    ClassifierConfig,
    ConfigError,
    ProviderConfig,
    ProxyConfig,
    ServerConfig,
)
from inference_proxy.main import create_app
from tests.conftest import chat_body, completion_json, make_config


async def request_with(config, handler, body=None):
    app = create_app(config, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client:
            return await client.post("/v1/chat/completions", json=body or chat_body(model="hardcore"))


def provider(base="http://p.test/v1", **kwargs):
    kwargs.setdefault("default_model", "big-model")
    return ProviderConfig(base_url=base, **kwargs)


def model_from(request):
    return json.loads(request.content).get("model")


async def test_tier_models_select_model_for_routed_tier():
    seen = []

    def handler(req):
        seen.append((req.url.path, model_from(req)))
        if req.url.host == "judge.test":
            return httpx.Response(200, json={"choices": [{"message": {"content": "2"}}]})
        return httpx.Response(200, json=completion_json("ok", model_from(req)))

    p = provider(tier_models={"trivial": "small-model"})
    config = make_config(providers={"p": p}, tiers={t: ["p"] for t in ("trivial", "moderate", "complex", "hardcore")})
    config = config.model_copy(update={"classifier": ClassifierConfig(base_url="http://judge.test/v1")})
    app = create_app(config, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as c:
            trivial = await c.post("/v1/chat/completions", json=chat_body())
            moderate = await c.post("/v1/chat/completions", json=chat_body(model="moderate"))
    assert trivial.status_code == moderate.status_code == 200
    assert trivial.json()["inference_proxy"]["model"] == "small-model"
    assert moderate.json()["inference_proxy"]["model"] == "big-model"
    assert [m for path, m in seen if path.endswith("chat/completions") and m not in {"local", "rater"}] == ["small-model", "big-model"]


async def test_missing_default_tries_chat_sibling_and_returns_headers():
    seen = []

    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "big-model"}, {"id": "mid-model"}, {"id": "text-embedding-3"}]})
        seen.append(model_from(req))
        if model_from(req) == "big-model":
            return httpx.Response(404, json={"error": {"message": "The model `big-model` does not exist"}})
        return httpx.Response(200, json=completion_json("ok", model_from(req)))

    response = await request_with(make_config(providers={"p": provider()}, tiers={"trivial": ["p"], "moderate": ["p"], "complex": ["p"], "hardcore": ["p"]}), handler)
    assert response.status_code == 200
    assert response.headers["X-Proxy-Provider"] == "p"
    assert response.headers["X-Proxy-Tier"] == "hardcore"
    assert seen == ["big-model", "mid-model"]
    assert response.json()["inference_proxy"]["model"] == "mid-model"


async def test_configured_fallback_models_precede_catalogue():
    seen, gets = [], []
    failed = False

    def handler(req):
        nonlocal failed
        if req.method == "GET":
            if failed:
                gets.append(req.url.path)
            return httpx.Response(200, json={"data": [{"id": "catalog-model"}]})
        seen.append(model_from(req))
        if model_from(req) == "big-model":
            failed = True
            return httpx.Response(404, json={"error": {"message": "model does not exist"}})
        return httpx.Response(200, json=completion_json("ok"))

    cfg = make_config(providers={"p": provider(fallback_models=["fallback-model"])}, tiers={t: ["p"] for t in ("trivial", "moderate", "complex", "hardcore")})
    response = await request_with(cfg, handler)
    assert response.status_code == 200
    assert seen == ["big-model", "fallback-model"]
    # Implementation issue: the retry path calls catalog.ensure before checking
    # configured fallbacks, so it still fetches /v1/models despite this config.
    assert "/v1/models" not in gets


async def test_pinned_model_does_not_fallback():
    seen, gets = [], []
    failed = False

    def handler(req):
        nonlocal failed
        if req.method == "GET":
            if failed:
                gets.append(req.url.path)
        else:
            seen.append(model_from(req))
            failed = True
        return httpx.Response(404, json={"error": {"message": "The model `big-model` does not exist"}})

    cfg = make_config(providers={"p": provider()}, tiers={t: ["p"] for t in ("trivial", "moderate", "complex", "hardcore")})
    response = await request_with(cfg, handler, chat_body(model="p:big-model"))
    assert response.status_code == 404
    assert response.json()["error"]["message"] == "The model `big-model` does not exist"
    assert seen == ["big-model"] and "/v1/models" not in gets


async def test_media_request_does_not_try_siblings():
    seen, gets = [], []

    def handler(req):
        if req.method == "GET":
            gets.append(1)
            return httpx.Response(200, json={"data": [{"id": "other-model"}]})
        seen.append(model_from(req))
        return httpx.Response(404, json={"error": {"message": "model does not exist"}})

    cfg = make_config(providers={"p": provider(capabilities=["text", "image"])}, tiers={t: ["p"] for t in ("trivial", "moderate", "complex", "hardcore")})
    body = chat_body(model="hardcore", messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}])
    response = await request_with(cfg, handler, body)
    assert response.status_code == 404
    assert seen == ["big-model"] and "/v1/models" not in gets


async def test_exhausted_siblings_move_to_next_provider():
    seen = []

    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "big-model"}, {"id": "mid-model"}]})
        name = req.url.host
        seen.append((name, model_from(req)))
        if name == "first.test":
            return httpx.Response(404, json={"error": {"message": "model does not exist"}})
        return httpx.Response(200, json=completion_json("next"))

    cfg = make_config(providers={"first": provider("http://first.test/v1"), "next": provider("http://next.test/v1")}, tiers={t: ["first", "next"] for t in ("trivial", "moderate", "complex", "hardcore")})
    response = await request_with(cfg, handler)
    assert response.status_code == 200
    assert response.headers["X-Proxy-Provider"] == "next"
    assert seen == [("first.test", "big-model"), ("first.test", "mid-model"), ("next.test", "big-model")]


async def test_models_refresh_catalogue_and_preserve_models_on_error():
    def handler(req):
        if req.url.host == "bad.test":
            return httpx.Response(503)
        return httpx.Response(200, json={"data": [{"id": "from-api"}]})

    cfg = make_config(providers={"good": provider(), "bad": provider("http://bad.test/v1", default_model="configured")}, tiers={t: ["good", "bad"] for t in ("trivial", "moderate", "complex", "hardcore")})
    app = create_app(cfg, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as c:
            assert (await c.post("/admin/api/models/refresh")).status_code == 200
            data = (await c.get("/admin/api/models")).json()["providers"]
    assert data["good"]["models"][0] == "from-api"
    assert data["good"]["fetched_at"] and "raw" not in data["good"]
    assert data["bad"]["error"] and "configured" in data["bad"]["models"]


def test_auto_tiers_and_empty_tier_validation():
    providers = {"p": {"base_url": "http://p/v1"}, "q": {"base_url": "http://q/v1"}}
    all_tiers = ProxyConfig.model_validate({"providers": providers})
    assert all(all_tiers.tiers[t] == ["p", "q"] for t in ("trivial", "moderate", "complex", "hardcore"))
    partial = ProxyConfig.model_validate({"providers": providers, "tiers": {"moderate": ["q"], "hardcore": ["p"]}})
    assert partial.tiers["trivial"] == ["q"] and partial.tiers["complex"] == ["p"]
    with pytest.raises((ConfigError, ValidationError)):
        ProxyConfig.model_validate({"providers": providers, "tiers": {"moderate": [], "hardcore": ["p"]}})


async def test_hosted_judge_uses_provider_url_model_and_key():
    seen = []

    def handler(req):
        seen.append((req.url.host, req.headers.get("Authorization"), model_from(req)))
        if req.url.host == "judge.test":
            return httpx.Response(200, json={"choices": [{"message": {"content": "2"}}]})
        return httpx.Response(200, json=completion_json("ok"))

    p = ProviderConfig(base_url="http://judge.test/v1", api_key="judge-key", default_model="judge-model")
    cfg = make_config(classifier=ClassifierConfig(provider="judge", base_url="http://local-judge/v1"), providers={"judge": p}, tiers={t: ["judge"] for t in ("trivial", "moderate", "complex", "hardcore")})
    app = create_app(cfg, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as c:
            response = await c.post("/v1/chat/completions", json=chat_body())
            note = (await c.get("/admin/api/models")).json()["judge"]
    assert response.status_code == 200
    assert ("judge.test", "Bearer judge-key", "judge-model") in seen
    assert note == "judge: provider judge"


async def test_auto_judge_fallbacks_on_connect_error_and_prefers_local_when_available():
    hosted = []

    def down(req):
        if req.url.host == "local.test":
            raise httpx.ConnectError("offline", request=req)
        hosted.append(req.url.host)
        if req.url.host == "hosted.test":
            return httpx.Response(200, json={"choices": [{"message": {"content": "2"}}]})
        return httpx.Response(200, json=completion_json("ok"))

    p = ProviderConfig(base_url="http://hosted.test/v1", api_key="key", default_model="judge-model")
    cfg = make_config(classifier=ClassifierConfig(provider="auto", base_url="http://local.test/v1"), providers={"hosted": p}, tiers={t: ["hosted"] for t in ("trivial", "moderate", "complex", "hardcore")})
    app = create_app(cfg, http_client=httpx.AsyncClient(transport=httpx.MockTransport(down)))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as c:
            response = await c.post("/v1/chat/completions", json=chat_body())
            note = (await c.get("/admin/api/models")).json()["judge"]
    assert response.status_code == 200 and "hosted.test" in hosted
    assert note == "judge: local server not reachable, using hosted"

    local_seen = []
    def up(req):
        local_seen.append(req.url.host)
        if req.url.host == "local.test":
            return httpx.Response(200, json={"choices": [{"message": {"content": "2"}}]})
        return httpx.Response(200, json=completion_json("ok"))
    cfg = cfg.model_copy(update={"classifier": cfg.classifier.model_copy(update={"base_url": "http://local.test/v1"})})
    app = create_app(cfg, http_client=httpx.AsyncClient(transport=httpx.MockTransport(up)))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as c:
            await c.post("/v1/chat/completions", json=chat_body())
            note = (await c.get("/admin/api/models")).json()["judge"]
    assert note == "judge: local server"


def install_claude(tmp_path, monkeypatch, *, stdout="cli reply", stderr="", code=0):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    script = bindir / "claude"
    script.write_text("#!/bin/sh\ncat >/dev/null\nprintf '%s' " + json.dumps(stdout) + "\nprintf '%s' " + json.dumps(stderr) + " >&2\nexit " + str(code) + "\n")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ.get('PATH', '')}")


async def test_claude_provider_chat_and_stream(tmp_path, monkeypatch):
    install_claude(tmp_path, monkeypatch)
    cfg = make_config(providers={"cc": ProviderConfig(type="claude", models=["sonnet"])}, tiers={"trivial": ["cc"], "moderate": ["cc"], "complex": ["cc"], "hardcore": ["cc"]})
    app = create_app(cfg, http_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(404))))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as c:
            # This currently exposes a proxy bug: the CLI's streamed byte response is
            # decoded as JSON without first being read by the non-streaming response path.
            normal = await c.post("/v1/chat/completions", json=chat_body(model="hardcore"))
            stream = await c.post("/v1/chat/completions", json=chat_body(model="hardcore", stream=True))
    assert normal.status_code == 200 and normal.json()["choices"][0]["message"]["content"] == "cli reply"
    assert normal.headers["X-Proxy-Provider"] == "cc"
    assert stream.status_code == 200 and stream.headers["content-type"].startswith("text/event-stream")
    assert "cli reply" in stream.text and "data: [DONE]" in stream.text


async def test_cli_quota_response_cools_account_and_uses_next_provider(tmp_path, monkeypatch):
    install_claude(tmp_path, monkeypatch, stdout="", stderr="usage limit reached", code=1)
    cfg = make_config(providers={"cc": ProviderConfig(type="claude", models=["sonnet"]), "next": provider("http://next.test/v1")}, tiers={t: ["cc", "next"] for t in ("trivial", "moderate", "complex", "hardcore")})
    def handler(req):
        return httpx.Response(200, json=completion_json("next"))
    app = create_app(cfg, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as c:
            response = await c.post("/v1/chat/completions", json=chat_body(model="hardcore"))
    assert response.status_code == 200 and response.headers["X-Proxy-Provider"] == "next"
    assert app.state.pools["cc"].accounts[0].cooldown_until > 0
