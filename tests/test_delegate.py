"""Coverage for request restrictions, CLI depth, and the MCP delegation surface."""
import json
import os
import stat

import httpx
import pytest

from inference_proxy import cli, mcp
from inference_proxy.config import ProviderConfig
from inference_proxy.main import create_app
from tests.conftest import chat_body, completion_json, make_config


def install_cli(tmp_path, monkeypatch, name="claude"):
    path = tmp_path / name
    path.write_text('#!/bin/sh\ncat >/dev/null\nprintf "%s" "$INFERENCE_PROXY_DEPTH"\n')
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")


async def post(cfg, body, handler):
    app = create_app(cfg, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as c:
            return await c.post("/v1/chat/completions", json=body)


def cli_api_config():
    ps = {"cc": ProviderConfig(type="claude", default_model="haiku"),
          "api": ProviderConfig(base_url="http://api.test/v1", default_model="m")}
    return make_config(providers=ps, tiers={t: ["cc", "api"] for t in ("trivial", "moderate", "complex", "hardcore")})


async def test_exclude_cli_pin_local_only_and_body_stripping(tmp_path, monkeypatch):
    install_cli(tmp_path, monkeypatch)
    seen = []
    def handler(req):
        data = json.loads(req.content); seen.append(data)
        return httpx.Response(200, json=completion_json("api", data["model"]))
    cfg = cli_api_config()
    r = await post(cfg, chat_body(model="hardcore"), handler)
    assert r.headers["X-Proxy-Provider"] == "cc"
    r = await post(cfg, chat_body(model="hardcore", inference_proxy={"exclude_types": ["claude"]}), handler)
    assert r.headers["X-Proxy-Provider"] == "api" and "inference_proxy" not in seen[-1]
    assert (await post(cfg, chat_body(model="cc", inference_proxy={"local_only": True, "exclude_types": ["claude"]}), handler)).headers["X-Proxy-Provider"] == "cc"
    assert (await post(cfg, chat_body(model="cc:haiku", inference_proxy={"local_only": True}), handler)).headers["X-Proxy-Provider"] == "cc"
    assert (await post(cfg, chat_body(model="hardcore", inference_proxy="ignore me"), handler)).headers["X-Proxy-Provider"] == "cc"


@pytest.mark.parametrize("url", ["http://192.168.2.3/v1", "http://127.0.0.1/v1", "http://localhost/v1"])
async def test_local_only_keeps_private_addresses(url):
    cfg = make_config(providers={"local": ProviderConfig(base_url=url, default_model="m"),
        "hosted": ProviderConfig(base_url="https://hosted.test/v1", default_model="m")},
        tiers={t: ["local", "hosted"] for t in ("trivial", "moderate", "complex", "hardcore")})
    seen=[]
    r=await post(cfg, chat_body(model="hardcore", inference_proxy={"local_only":True}), lambda req:(seen.append(req.url.host) or httpx.Response(200,json=completion_json("ok"))))
    assert r.status_code == 200 and url.split("//")[1].split("/")[0] in seen
    only=make_config(providers={"hosted":ProviderConfig(base_url="https://hosted.test/v1",default_model="m"), "cc":ProviderConfig(type="claude")},tiers={t:["hosted","cc"] for t in ("trivial","moderate","complex","hardcore")})
    r=await post(only,chat_body(model="hardcore",inference_proxy={"local_only":True}),lambda req:httpx.Response(200,json=completion_json("no")))
    assert r.status_code==400 and "restrictions" in r.json()["error"]["message"]


async def test_cost_metadata_only_when_price_and_usage():
    cfg=make_config(providers={"p":ProviderConfig(base_url="http://p.test/v1",default_model="m")},tiers={t:["p"] for t in ("trivial","moderate","complex","hardcore")})
    omit_usage=False
    def handler(req):
        data=completion_json("ok", "m")
        if omit_usage: data.pop("usage")
        return httpx.Response(200,json=data)
    app=create_app(cfg,http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    async with app.router.lifespan_context(app):
        app.state.prices.set_manual("p","m",1,2)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://proxy") as c:
            r=await c.post("/v1/chat/completions",json=chat_body(model="hardcore"))
            assert r.json()["inference_proxy"]["cost_usd"] == pytest.approx(0.000003)
            app.state.prices.db.execute("delete from prices"); app.state.prices.db.commit()
            r=await c.post("/v1/chat/completions",json=chat_body(model="hardcore"))
            assert "cost_usd" not in r.json()["inference_proxy"]
            app.state.prices.set_manual("p","m",1,2)
            omit_usage=True
            r=await c.post("/v1/chat/completions",json=chat_body(model="hardcore"))
            assert "cost_usd" not in r.json()["inference_proxy"]


def test_cli_depth_and_command(monkeypatch):
    for value, expected in [(None,"1"),("2","3"),("x","1"),("","1")]:
        if value is None: monkeypatch.delenv("INFERENCE_PROXY_DEPTH",raising=False)
        else: monkeypatch.setenv("INFERENCE_PROXY_DEPTH",value)
        # Known implementation bug: child_env currently raises ValueError for non-numeric values.
        assert cli.child_env()["INFERENCE_PROXY_DEPTH"] == expected
    assert "--strict-mcp-config" in cli.command("claude",None,[]) and "-p" in cli.command("claude",None,[])
    assert "--ephemeral" in cli.command("codex",None,[])


@pytest.mark.parametrize("name",["claude","codex"])
async def test_run_cli_passes_depth(tmp_path,monkeypatch,name):
    install_cli(tmp_path,monkeypatch,name)
    monkeypatch.setenv("INFERENCE_PROXY_DEPTH","2")
    response=await cli.run_cli(name,None,{"messages":[{"role":"user","content":"hi"}]},5,False)
    await response.aread()
    assert response.status_code==200 and json.loads(response.content)["choices"][0]["message"]["content"]=="3"


def proxy_for(handler, monkeypatch, key=None):
    monkeypatch.setenv("INFERENCE_PROXY_URL","http://proxy.test")
    if key: monkeypatch.setenv("INFERENCE_PROXY_API_KEY",key)
    else: monkeypatch.delenv("INFERENCE_PROXY_API_KEY",raising=False)
    return mcp.Proxy(httpx.Client(transport=httpx.MockTransport(handler)))


def tool(proxy,name,args=None,ident=1):
    return mcp.handle({"jsonrpc":"2.0","id":ident,"method":"tools/call","params":{"name":name,"arguments":args or {}}},proxy)["result"]


def payload(result): return json.loads(result["content"][0]["text"])


def test_mcp_protocol_and_tool_schemas(monkeypatch):
    p=proxy_for(lambda req:httpx.Response(200,json={}),monkeypatch)
    assert mcp.handle({"id":1,"method":"initialize","params":{"protocolVersion":"v"}},p)["result"]["protocolVersion"]=="v"
    assert mcp.handle({"method":"notifications/initialized"},p) is None
    assert mcp.handle({"id":2,"method":"ping"},p)["result"]=={}
    assert mcp.handle({"id":3,"method":"wat"},p)["error"]["code"]==-32601
    tools=mcp.handle({"id":4,"method":"tools/list"},p)["result"]["tools"]
    assert [t["name"] for t in tools]==["delegate","delegate_status","delegate_ladder","delegate_usage"]
    assert all(t["inputSchema"]["type"]=="object" for t in tools)


def test_delegate_request_and_errors(monkeypatch):
    seen=[]
    fail=False
    def handler(req):
        nonlocal fail
        seen.append(req)
        if req.url.path.endswith("completions"):
            if fail: return httpx.Response(502,json={"error":{"message":"all upstream providers failed"}})
            return httpx.Response(200,json={"choices":[{"message":{"content":"answer"}}],"usage":{"total_tokens":2},"inference_proxy":{"tier":"moderate","provider":"p","model":"m","difficulty":4,"cost_usd":.2}})
        return httpx.Response(200,json={})
    p=proxy_for(handler,monkeypatch,"secret")
    result=payload(tool(p,"delegate",{"prompt":"do","system":"sys","tier":"auto","max_tokens":5,"temperature":.2,"local_only":True}))
    body=json.loads(seen[-1].content)
    assert body["messages"]==[{"role":"system","content":"sys"},{"role":"user","content":"do"}]
    assert body["model"]=="auto" and body["max_tokens"]==5 and body["temperature"]==.2
    assert body["inference_proxy"]=={"exclude_types":["claude","codex"],"local_only":True}
    assert seen[-1].headers["Authorization"]=="Bearer secret"
    assert result=={"text":"answer","tier":"moderate","provider":"p","model":"m","difficulty":4,"usage":{"total_tokens":2},"cost_usd":.2}
    tool(p,"delegate",{"prompt":"x","allow_cli":True})
    assert "exclude_types" not in json.loads(seen[-1].content)["inference_proxy"]
    assert "max_tokens" not in json.loads(seen[-1].content)
    fail=True
    result=tool(p,"delegate",{"prompt":"x"})
    assert result["isError"] and "all upstream providers failed" in result["content"][0]["text"]
    monkeypatch.setenv("INFERENCE_PROXY_DEPTH","2")
    result=tool(p,"delegate",{"prompt":"x"})
    assert result["isError"] and "depth limit" in result["content"][0]["text"]
    monkeypatch.setenv("INFERENCE_PROXY_DEPTH","1")
    fail=False
    assert payload(tool(p,"delegate",{"prompt":"x"}))["text"] == "answer"
    assert tool(p,"missing")["isError"]


def test_status_ladder_usage(monkeypatch):
    calls=[]
    def handler(req):
        calls.append(req)
        if req.url.path.endswith("pools"): return httpx.Response(200,json={"pools":1})
        if req.url.path.endswith("models"): return httpx.Response(200,json={"pins":[1],"judge":"local"})
        if req.url.path.endswith("ladder"): return httpx.Response(200,json={"rungs":[{"rung":1,"difficulty_from":2,"difficulty_to":3,"provider":"p","model":"m","score":9,"extra":0}],"tiers":{},"unscored":[1,2]})
        return httpx.Response(200,json={"days":7})
    p=proxy_for(handler,monkeypatch)
    assert payload(tool(p,"delegate_status"))=={"pools":{"pools":1},"pinned_endpoints":[1],"judge":"local"}
    assert payload(tool(p,"delegate_ladder",{"rungs":2,"tolerance":.1}))=={"rungs":[{"rung":1,"difficulty_from":2,"difficulty_to":3,"provider":"p","model":"m","score":9}],"tiers":{},"unscored":2}
    req=calls[-1]; assert dict(req.url.params)=={"rungs":"2","tolerance":"0.1"}
    tool(p,"delegate_usage",{"days":4}); assert calls[-1].url.params["days"]=="4"
    monkeypatch.setenv("INFERENCE_PROXY_DEPTH","2")
    assert "pools" in payload(tool(p,"delegate_status"))
