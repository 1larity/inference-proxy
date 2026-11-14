"""Price proposals and request-cost accounting."""

import json
import time

import httpx
import pytest

from inference_proxy.config import ProviderConfig, ServerConfig
from inference_proxy.main import create_app
from inference_proxy.pricing import PriceStore, extract_price
from tests.conftest import SSEStream, chat_body, completion_json, make_config


def test_extract_price():
    raw = {"pricing": {"prompt": "0.0000003", "completion": "0.0000012"}, "context_length": 131072}
    assert extract_price(raw) == (0.3, 1.2, 131072)
    for bad in (
        {"pricing": {"prompt": "-0.1", "completion": "0.2"}},
        {"pricing": {"prompt": "0.1"}},
        {"pricing": {"prompt": "nope", "completion": "0.2"}},
        None,
        [],
    ):
        assert extract_price(bad) is None


def test_price_store_proposals_apply_reject_and_manual():
    store = PriceStore()
    assert store.propose("p", "m", (2, 4, 100), "test") == "new"
    proposal = store.listing("proposed")[0]
    assert store.active("p", "m") is None
    assert store.apply([proposal["id"]]) == 1
    active = store.active("p", "m")
    fetched = active["fetched_at"]
    assert store.propose("p", "m", (2, 4, 100), "test") is None
    assert store.active("p", "m")["fetched_at"] >= fetched
    assert store.propose("p", "m", (3, 5, 200), "test") == "changed"
    changed = store.listing("proposed")[0]
    assert (changed["old_input"], changed["old_output"]) == (2, 4)
    assert store.propose("p", "unknown", None, "test") is None
    assert store.propose("p", "m", None, "test") == "retired"
    retirement = store.listing("proposed")[0]
    assert store.apply([retirement["id"]]) == 1
    assert store.active("p", "m") is None

    store.propose("p", "a", (1, 1, None), "test")
    store.propose("p", "b", (2, 2, None), "test")
    ids = {r["model"]: r["id"] for r in store.listing("proposed")}
    assert store.apply([ids["a"]]) == 1
    assert store.active("p", "a") is not None and store.active("p", "b") is None
    assert store.reject() == 1 and store.active("p", "a") is not None
    store.propose("p", "a", (7, 8, None), "test")
    assert store.reject() == 1 and store.active("p", "a")["input_per_m"] == 1
    store.set_manual("p", "a", 9, 10)
    assert store.active("p", "a")["input_per_m"] == 9
    with pytest.raises(ValueError):
        store.set_manual("p", "a", -1, 0)


def test_ledger_cost_snapshot_unpriced_days_and_order():
    store = PriceStore()
    store.set_manual("p", "m", 2, 4)
    assert store.record_usage("p", "m", "complex", {"prompt_tokens": 1000, "completion_tokens": 500}) == pytest.approx(.004)
    assert store.record_usage("p", "unknown", "trivial", {"prompt_tokens": 1}) is None
    store.set_manual("p", "m", 100, 100)
    store.record_usage("p", "m", "hardcore", {"prompt_tokens": 1000, "completion_tokens": 0})
    store.db.execute("insert into usage(ts,provider,model,tier,input_tokens,output_tokens,cost_usd) values(?,?,?,?,?,?,?)", (time.time() - 30 * 86400, "old", "m", "x", 1, 1, 100))
    rows = store.usage_summary(7)
    assert all(r["provider"] != "old" for r in rows)
    assert rows[0]["tier"] == "hardcore" and rows[0]["cost_usd"] == pytest.approx(.1)
    unpriced = next(r for r in rows if r["model"] == "unknown")
    assert unpriced["unpriced"] == 1 and unpriced["cost_usd"] is None
    assert next(r for r in rows if r["tier"] == "complex")["cost_usd"] == pytest.approx(.004)


MODELS = {"data": [
    {"id": "big-model", "pricing": {"prompt": "0.000002", "completion": "0.000008"}, "context_length": 64000},
    {"id": "other-model", "pricing": {"prompt": "0.1", "completion": "0.2"}},
]}


async def with_app(handler, config=None, run=None):
    app = create_app(config or make_config(), http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client:
            return await run(client, app)


async def test_price_endpoints_refresh_apply_reject_manual_and_stale():
    models = MODELS

    def handler(req):
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json=models)
        return httpx.Response(200, json=completion_json("ok"))

    cfg = make_config(providers={"p": ProviderConfig(base_url="http://p.test/v1", default_model="big-model", tier_models={"trivial": "big-model"}, fallback_models=["other-model"])}, tiers={t: ["p"] for t in ("trivial", "moderate", "complex", "hardcore")})

    async def run(c, app):
        r = await c.post("/admin/api/prices/refresh")
        assert {x["model"] for x in r.json()["changes"]} == {"big-model", "other-model"}
        assert {x["status"] for x in r.json()["prices"]} == {"proposed"}
        all_response = await c.post("/admin/api/prices/refresh", json={"all": True})
        assert {x["model"] for x in all_response.json()["changes"]} == {"big-model", "other-model"}
        # An unrelated model is proposed when all=true.
        models["data"].append({"id": "third", "pricing": {"prompt": "0.3", "completion": "0.4"}})
        all_response = await c.post("/admin/api/prices/refresh", json={"all": True})
        assert any(x["model"] == "third" for x in all_response.json()["changes"])
        third = next(x for x in all_response.json()["prices"] if x["model"] == "third")
        applied = await c.post("/admin/api/prices/apply", json={"ids": [third["id"]]})
        assert applied.json()["applied"] == 1
        assert sum(row["status"] == "active" for row in applied.json()["prices"]) == 1
        assert (await c.post("/admin/api/prices/reject", json={})).json()["rejected"] == 2
        assert (await c.post("/admin/api/prices/refresh")).status_code == 200
        assert (await c.post("/admin/api/prices/apply", json={})).json()["applied"] == 2
        assert (await c.put("/admin/api/prices", json={"provider": "p", "model": "manual", "input_per_m": -1, "output_per_m": 2})).status_code == 400
        assert (await c.put("/admin/api/prices", json={"provider": "p", "model": "bad"})).status_code == 400
        await c.put("/admin/api/prices", json={"provider": "p", "model": "manual", "input_per_m": 1, "output_per_m": 2})
        app.state.prices.db.execute("update prices set fetched_at=? where model='manual'", (time.time() - 30 * 86400,))
        view = (await c.get("/admin/api/prices")).json()
        manual = next(x for x in view["prices"] if x["model"] == "manual")
        assert manual["age_days"] >= 29 and manual["stale"] is True
        models["data"][0]["pricing"]["prompt"] = "0.000003"
        changed = await c.post("/admin/api/prices/refresh")
        assert next(x for x in changed.json()["changes"] if x["model"] == "big-model")["kind"] == "changed"
        models["data"] = [x for x in models["data"] if x["id"] != "big-model"]
        retired = await c.post("/admin/api/prices/refresh")
        assert next(x for x in retired.json()["changes"] if x["model"] == "big-model")["kind"] == "retired"
    await with_app(handler, cfg, run)


async def test_refresh_skips_cli_and_failed_models():
    calls = []
    def handler(req):
        calls.append(req.url.host)
        if req.url.host == "bad.test":
            return httpx.Response(503)
        return httpx.Response(200, json=MODELS)
    cfg = make_config(providers={"cli": ProviderConfig(type="claude", base_url="http://cli.test/v1", default_model="c"), "bad": ProviderConfig(base_url="http://bad.test/v1", default_model="m")}, tiers={t: ["cli", "bad"] for t in ("trivial", "moderate", "complex", "hardcore")})
    async def run(c, app):
        assert (await c.post("/admin/api/prices/refresh")).status_code == 200
        assert calls and all(p == "bad.test" for p in calls if p == "bad.test")
        assert "cli.test" not in calls
    await with_app(handler, cfg, run)


async def test_proxy_usage_json_stream_missing_and_failed():
    def handler(req):
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json=MODELS)
        body = json.loads(req.content)
        if body.get("model") == "fail":
            return httpx.Response(500, json={"error": {"message": "bad"}})
        if body.get("stream"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=SSEStream(b'data: {"usage":{"prompt_tokens":1000,"completion_tokens":500}}\n\n', b'data: [DONE]\n\n'))
        result = completion_json("ok", body.get("model"))
        if body.get("model") == "no-usage":
            result.pop("usage")
        else:
            result["usage"] = {"prompt_tokens": 1000, "completion_tokens": 500}
        return httpx.Response(200, json=result)
    cfg = make_config(providers={"p": ProviderConfig(base_url="http://p.test/v1", default_model="big-model", api_key="key")}, tiers={t: ["p"] for t in ("trivial", "moderate", "complex", "hardcore")})
    async def run(c, app):
        await c.put("/admin/api/prices", json={"provider": "p", "model": "big-model", "input_per_m": 2, "output_per_m": 4})
        await c.post("/v1/chat/completions", json=chat_body(model="p:big-model"))
        await c.post("/v1/chat/completions", json=chat_body(model="p:big-model", stream=True))
        await c.post("/v1/chat/completions", json=chat_body(model="p:no-usage"))
        await c.post("/v1/chat/completions", json=chat_body(model="p:fail"))
        data = (await c.get("/admin/api/usage")).json()
        assert data["total_cost_usd"] == pytest.approx(.008)
        assert sum(x["requests"] for x in data["usage"]) == 2
        assert sum(x["cost_usd"] or 0 for x in data["usage"]) == pytest.approx(.008)
    await with_app(handler, cfg, run)


async def test_pricing_routes_require_bearer_auth():
    cfg = make_config(server=ServerConfig(settings_file=None, api_key="secret"))
    async def handler(req):
        return httpx.Response(200, json=MODELS)
    async def run(c, app):
        for method, url, kwargs in [
            ("get", "/admin/api/prices", {}), ("get", "/admin/api/usage", {}),
            ("post", "/admin/api/prices/refresh", {}), ("post", "/admin/api/prices/apply", {"json": {}}),
            ("post", "/admin/api/prices/reject", {"json": {}}), ("put", "/admin/api/prices", {"json": {}}),
        ]:
            assert (await getattr(c, method)(url, **kwargs)).status_code == 401
    await with_app(handler, cfg, run)
