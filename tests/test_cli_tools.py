"""Requests that define tools must never reach a CLI provider, which cannot return tool calls."""

import json
import os
import stat

import httpx

from inference_proxy.config import ProviderConfig
from inference_proxy.main import create_app
from tests.conftest import chat_body, completion_json, make_config

TOOLS = [{"type": "function", "function": {"name": "grep", "parameters": {"type": "object", "properties": {}}}}]


def fake_claude(tmp_path, monkeypatch):
    script = tmp_path / "claude"
    script.write_text("#!/bin/sh\ncat >/dev/null\necho from-cli\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")


async def post(config, body, seen):
    def handler(req):
        seen.append(json.loads(req.content))
        return httpx.Response(200, json=completion_json("from-api", "m"))

    app = create_app(config, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client:
            return await client.post("/v1/chat/completions", json=body)


def config(tiers):
    providers = {
        "cc": ProviderConfig(type="claude", default_model="haiku"),
        "api": ProviderConfig(base_url="http://api.test/v1", default_model="m"),
    }
    return make_config(providers=providers, tiers={t: tiers for t in ("trivial", "moderate", "complex", "hardcore")})


async def test_tools_skip_the_cli_and_use_the_api_provider(tmp_path, monkeypatch):
    fake_claude(tmp_path, monkeypatch)
    seen = []
    r = await post(config(["cc", "api"]), chat_body(model="hardcore") | {"tools": TOOLS}, seen)
    assert r.status_code == 200 and r.headers["X-Proxy-Provider"] == "api"
    assert seen and seen[0]["tools"] == TOOLS


async def test_without_tools_the_cli_is_still_first(tmp_path, monkeypatch):
    fake_claude(tmp_path, monkeypatch)
    r = await post(config(["cc", "api"]), chat_body(model="hardcore"), [])
    assert r.status_code == 200 and r.headers["X-Proxy-Provider"] == "cc"


async def test_legacy_functions_field_counts_as_tools(tmp_path, monkeypatch):
    fake_claude(tmp_path, monkeypatch)
    r = await post(config(["cc", "api"]), chat_body(model="hardcore") | {"functions": [{"name": "grep"}]}, [])
    assert r.headers["X-Proxy-Provider"] == "api"


async def test_only_cli_providers_give_a_clear_error(tmp_path, monkeypatch):
    fake_claude(tmp_path, monkeypatch)
    cfg = make_config(providers={"cc": ProviderConfig(type="claude", default_model="haiku")},
                      tiers={t: ["cc"] for t in ("trivial", "moderate", "complex", "hardcore")})
    r = await post(cfg, chat_body(model="hardcore") | {"tools": TOOLS}, [])
    assert r.status_code == 400 and "tool" in r.json()["error"]["message"]


async def test_pinning_the_cli_is_still_honoured(tmp_path, monkeypatch):
    fake_claude(tmp_path, monkeypatch)
    r = await post(config(["cc", "api"]), chat_body(model="cc") | {"tools": TOOLS}, [])
    assert r.status_code == 200 and r.headers["X-Proxy-Provider"] == "cc"
