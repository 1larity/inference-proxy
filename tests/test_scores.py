"""Score ingestion, ladder suggestions and opt-in price research."""
import io
import json
import os
import zipfile

import httpx
import pytest

from inference_proxy.benchmarks import (ScoreStore, candidate_keys, fetch_aa, fetch_epoch,
                                        normalise, parse_aa, parse_epoch)
from inference_proxy.cli import ask_cli
from inference_proxy.config import ProviderConfig, ResearchConfig
from inference_proxy.ladder import suggest
from inference_proxy.main import create_app
from inference_proxy.pricing import PriceStore
from inference_proxy.research import build_prompt, validate
from tests.conftest import make_config


HEADER = "Model,Display name,eci,eci_ci_low,eci_ci_high,date,Organization,Country (of organization),Model accessibility,Accessibility group,model_versions\n"


def epoch_zip(csv=HEADER + "glm-5.3-flash,GLM Flash,42,40,44,2026-01-01,ZAI,CN,Open,open,1\nqwen,Qwen,0.5,0.4,0.6,2026-01-02,A,CN,Open,open,1\nbad,Bad,nope,,,,,,,,\n"):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("epoch_capabilities_index/eci_scores.csv", csv)
    return out.getvalue()


def aa_page(score=58.1):
    return {"intelligence_index_version": "x", "pagination": {"has_more": False}, "data": [
        {"name": "Big", "slug": "org/big-model", "model_creator": {"name": "Org"},
         "evaluations": {"artificial_analysis_intelligence_index": score},
         "pricing": {"price_1m_input_tokens": .5, "price_1m_output_tokens": 2.0},
         "performance": {"median_output_tokens_per_second": 120}},
        {"name": "skip", "slug": "skip", "evaluations": {"artificial_analysis_intelligence_index": None}}]}


def test_normalise_and_candidate_keys():
    for n in ("GLM-5.3-Flash", "z-ai/GLM-5.3-Flash", "glm_5.3 flash"):
        assert normalise(n) == "glm-5.3-flash"
    assert "qwen3.5-9b" in candidate_keys("Qwen3.5-9B-GGUF")


def test_epoch_parser():
    rows = parse_epoch(epoch_zip())
    assert [r["name"] for r in rows] == ["GLM Flash", "Qwen"]
    assert rows[0]["score"] == 42 and "glm-5.3-flash" in rows[0]["keys"]
    assert rows[0]["detail"] == {"low": "40", "high": "44", "date": "2026-01-01", "organization": "ZAI", "accessibility": "Open"}
    with pytest.raises(ValueError):
        parse_epoch(bad_zip())


def bad_zip():
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z: z.writestr("missing.csv", "x")
    return b.getvalue()


def test_aa_parser_and_store():
    rows = parse_aa([aa_page()])
    assert len(rows) == 1 and rows[0]["score"] == 58.1
    assert rows[0]["detail"] == {"creator": "Org", "input_per_m": .5, "output_per_m": 2., "tokens_per_s": 120, "version": "x"}
    store = ScoreStore(PriceStore(None).db)
    epoch = [{"name": "one", "keys": ["glm-5.3-flash"], "score": 1, "detail": {}}]
    aa = [{"name": "two", "keys": ["glm-5.3-flash"], "score": 2, "detail": {}}]
    store.replace("epoch", epoch); store.replace("aa", aa)
    assert store.lookup("z-ai/GLM-5.3-Flash")["score"] == 1
    assert store.lookup("GLM-5.3-Flash", "aa")["score"] == 2
    assert store.lookup("Qwen3.5-9B-GGUF") is None
    store.replace("epoch", [{"name": "q", "keys": ["qwen"], "score": 3, "detail": {}}])
    assert store.lookup("glm-5.3-flash", "aa")["score"] == 2
    assert store.summary()["epoch"]["models"] == store.summary()["aa"]["models"] == 1


@pytest.mark.asyncio
async def test_fetch_sources():
    seen = []
    def handler(req):
        seen.append(req)
        if req.url.host == "aa.test":
            page = {**aa_page(), "pagination": {"has_more": True}} if req.url.params["page"] == "1" else {**aa_page(61), "pagination": {"has_more": False}}
            if req.url.params["page"] == "2": page["data"][0]["slug"] = "org/second-model"
            return httpx.Response(200, json=page)
        return httpx.Response(200, content=epoch_zip())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        # URLs are replaced in the module constants to keep the implementation under test unchanged.
        import inference_proxy.benchmarks as b
        old_aa, old_epoch = b.AA_URL, b.EPOCH_URL
        b.AA_URL, b.EPOCH_URL = "https://aa.test/models", "https://epoch.test/data"
        try:
            aa = await fetch_aa(client, "secret")
            epoch = await fetch_epoch(client)
        finally:
            b.AA_URL, b.EPOCH_URL = old_aa, old_epoch
    assert len(aa) == 2 and len(epoch) == 2
    assert [r.headers["x-api-key"] for r in seen if r.url.host == "aa.test"] == ["secret", "secret"]
    assert [r.url.params["page"] for r in seen if r.url.host == "aa.test"] == ["1", "2"]


def c(model, score, cost=0, speed=0):
    return {"provider": "p", "model": model, "score": score, "input_per_m": cost, "output_per_m": 0, "tokens_per_s": speed}


def test_ladder_quantiles_tolerance_and_tiers():
    rows = [c(str(i), i * 10) for i in range(1, 7)]
    got = suggest(rows, 3)
    assert len(got["rungs"]) == 3
    assert [(r["difficulty_from"], r["difficulty_to"]) for r in got["rungs"]] == [(1, 3), (4, 7), (8, 10)]
    assert got["rungs"][0]["difficulty_from"] == 1 and got["rungs"][-1]["difficulty_to"] == 10
    assert len(suggest(rows, 99)["rungs"]) == 10 and len(suggest(rows, 0)["rungs"]) == 1
    assert suggest(rows + [{"provider": "p", "model": "u", "score": None}], 3)["unscored"][0]["model"] == "u"
    close = [c("strong", 100, 5, 100), c("cheap", 98, 1, 50), c("fast", 99, 1, 90)]
    assert suggest(close, 1, .03)["rungs"][0]["model"] == "fast"
    assert suggest([c("a", 100, 1), c("b", 101, 1)], 1, .03)["rungs"][0]["model"] == "b"
    assert suggest([c("a", 1), c("b", 100)], 2, 0)["rungs"][0]["model"] == "a"
    bands = suggest([c(str(i), i * 10) for i in range(1, 11)], 10)["tiers"]
    assert [bands[t]["rung"] for t in ("trivial", "moderate", "complex", "hardcore")] == [1, 4, 7, 9]
    assert suggest([], 3)["rungs"] == []


def test_research_validation_and_prompt():
    valid = {"model": "BIG-MODEL", "input_per_m": 1, "output_per_m": 2, "source_url": "https://example.com"}
    bad = [None, {**valid, "model": "unknown"}, {**valid, "source_url": "http://example.com"},
           {**valid, "input_per_m": -1}, {**valid, "input_per_m": 1001}, {**valid, "input_per_m": "1"}, {**valid, "input_per_m": True}]
    assert validate({"prices": [valid, *bad]}, ["big-model"]) == [{"model": "big-model", "input_per_m": 1., "output_per_m": 2., "source_url": "https://example.com"}]
    assert validate([valid], ["big-model"]) == []
    prompt = build_prompt(["a", "b"], "https://example.com")
    assert "a, b" in prompt and "https://example.com" in prompt


@pytest.mark.asyncio
async def test_ask_cli_web_guards(tmp_path, monkeypatch):
    with pytest.raises(ValueError): await ask_cli("codex", "x", web=True)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(RuntimeError): await ask_cli("claude", "x")


async def app_call(handler, cfg, run):
    app = create_app(cfg, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client:
            return await run(client, app)


@pytest.mark.asyncio
async def test_score_endpoints():
    import inference_proxy.main as main
    old = main.fetch_epoch
    async def parsed(_): return parse_epoch(epoch_zip())
    main.fetch_epoch = parsed
    cfg = make_config()
    try:
        async def run(c, app):
            assert (await c.post("/admin/api/scores/refresh", json={"source": "epoch"})).json()["result"]["epoch"] == 2
            data = (await c.get("/admin/api/scores")).json()["sources"]["epoch"]
            assert data["models"] == 3 and "CC BY 4.0" in data["credit"]
            assert "skipped" in (await c.post("/admin/api/scores/refresh", json={"source": "aa"})).json()["result"]["aa"]
            assert (await c.post("/admin/api/scores/refresh", json={"source": "bad"})).status_code == 400
        await app_call(lambda r: httpx.Response(200), cfg, run)
    finally: main.fetch_epoch = old


@pytest.mark.asyncio
async def test_score_refresh_reports_upstream_failure(monkeypatch):
    import inference_proxy.main as main
    async def failed(_): raise httpx.HTTPStatusError("upstream down", request=httpx.Request("GET", "https://epoch.test"), response=httpx.Response(503))
    monkeypatch.setattr(main, "fetch_epoch", failed)
    async def run(c, app):
        response = await c.post("/admin/api/scores/refresh", json={"source": "epoch"})
        assert response.status_code == 200 and response.json()["result"]["epoch"].startswith("failed:")
    await app_call(lambda req: httpx.Response(200), make_config(), run)


@pytest.mark.asyncio
async def test_ladder_endpoint_and_sources():
    models = {"data": [{"id": "big-model"}, {"id": "small-model"}]}
    cfg = make_config(providers={"a": ProviderConfig(base_url="http://p.test/v1", default_model="big-model"),
                                 "b": ProviderConfig(base_url="http://p.test/v1", default_model="big-model")},
                      tiers={t: ["a", "b"] for t in ("trivial", "moderate", "complex", "hardcore")})
    def handler(req): return httpx.Response(200, json=models)
    async def run(c, app):
        app.state.scores.replace("epoch", [{"name": "Big", "keys": ["big-model"], "score": 80, "detail": {}}])
        app.state.scores.replace("aa", [{"name": "Small", "keys": ["small-model"], "score": 20, "detail": {}}])
        data = (await c.get("/admin/api/ladder?rungs=1&source=epoch")).json()
        assert len(data["rungs"]) == 1 and data["rungs"][0]["score"] == 80
        assert data["suggested_tier_models"] and data["sources"]["epoch"]["models"] == 1
        assert (await c.get("/admin/api/ladder?source=nope")).status_code == 400
        # Both providers share one endpoint and model, so only one candidate is returned.
        assert len([x for x in data["unscored"] if x["model"] == "big-model"]) == 0
    await app_call(handler, cfg, run)


@pytest.mark.asyncio
async def test_research_endpoint_proposals_and_cli(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"; bindir.mkdir()
    argv = tmp_path / "argv"
    output = {"prices": [{"model": "big-model", "input_per_m": 1.5, "output_per_m": 6, "source_url": "https://example.com/pricing"},
                         {"model": "missing", "input_per_m": 1, "output_per_m": 2, "source_url": "https://example.com"},
                         {"model": "other-model", "input_per_m": 1, "output_per_m": 2, "source_url": "http://example.com"}]}
    script = bindir / "claude"
    script.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > '+str(argv)+'\ncat >/dev/null\nprintf %s '+json.dumps(json.dumps(output))+'\n')
    script.chmod(0o755); monkeypatch.setenv("PATH", f"{bindir}:{os.environ.get('PATH','')}")
    cfg = make_config(providers={"p": ProviderConfig(base_url="http://p.test/v1", default_model="big-model"),
                                 "cc": ProviderConfig(type="claude", models=["sonnet"]),
                                 "cx": ProviderConfig(type="codex", models=["codex"]),
                                 "wrong": ProviderConfig(type="claude", models=["sonnet"])},
                      tiers={t: ["p"] for t in ("trivial", "moderate", "complex", "hardcore")})
    async def run(c, app):
        def resp(req): return httpx.Response(200, json={"data": [{"id": "big-model"}, {"id": "other-model"}]})
        # app transport is configured by app_call; the handler below handles /models.
        for cli in ("cx", "p"):
            r = await c.post("/admin/api/research/prices", json={"provider": "p", "cli_provider": cli})
            assert r.status_code == 400
        r = await c.post("/admin/api/research/prices", json={"provider": "p", "cli_provider": "cc", "url": "https://example.com/pricing"})
        assert r.status_code == 200 and r.json()["read"] == 1
        proposal = app.state.prices.listing("proposed")
        assert len(proposal) == 1 and proposal[0]["source"] == "web research" and proposal[0]["status"] == "proposed"
        assert app.state.prices.active("p", "big-model") is None
        args = argv.read_text().splitlines(); assert "--allowedTools" in args and "WebSearch" in args and "WebFetch" in args
    await app_call(lambda req: httpx.Response(200, json={"data": [{"id": "big-model"}, {"id": "other-model"}]}), cfg, run)


@pytest.mark.asyncio
async def test_research_garbage_is_400(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"; bindir.mkdir()
    script = bindir / "claude"; script.write_text("#!/bin/sh\ncat >/dev/null\necho garbage\n"); script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ.get('PATH','')}")
    cfg = make_config(providers={"p": ProviderConfig(base_url="http://p.test/v1", default_model="big-model"), "cc": ProviderConfig(type="claude", models=["sonnet"])},
                      tiers={t: ["p"] for t in ("trivial", "moderate", "complex", "hardcore")})
    async def run(c, app):
        r = await c.post("/admin/api/research/prices", json={"provider": "p", "cli_provider": "cc"})
        # Current handler treats unparseable CLI output as an empty proposal set (200).
        # Requirement is to reject malformed output with 400; this assertion intentionally exposes that bug.
        assert r.status_code == 400 and "error" in r.json()
    await app_call(lambda req: httpx.Response(200, json={"data": [{"id": "big-model"}]}), cfg, run)


@pytest.mark.asyncio
async def test_research_scheduler_waits_for_due_interval():
    calls = []
    cfg = make_config(research=ResearchConfig(enabled=True, every_days=7))
    def handler(req):
        calls.append((req.url.host, req.url.path))
        return httpx.Response(200, json={"data": []})
    app = create_app(cfg, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    async with app.router.lifespan_context(app):
        app.state.prices.meta_set("research_last_run", "9999999999")
        # Lifespan starts the task before test setup, so set the timestamp synchronously
        # before yielding to the event loop via the initial loop's first scheduling point.
        await __import__("asyncio").sleep(0)
        assert app.state.prices.meta_get("research_last_run") == "9999999999"
        assert not any(host == "local.test" and path.endswith("/models") for host, path in calls)
