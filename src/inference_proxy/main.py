"""FastAPI application exposing an OpenAI-compatible, difficulty-routing proxy."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

if __package__ in (None, ""):
    # Executed directly (python src/inference_proxy/main.py): re-import through
    # the package so relative imports resolve, then start the server.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from inference_proxy.main import run

    raise SystemExit(run())

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import ValidationError
from starlette.background import BackgroundTask

from .classifier import classify_difficulty
from .benchmarks import SOURCES, ScoreStore, fetch_aa, fetch_epoch
from .cli import CLI_TYPES, ask_cli, run_cli
from .config import ConfigError, ProxyConfig, compute_pins, endpoint_key, load_config, load_config_dict
from .context import ContextRegistry, probe_model_context
from .media import media_capabilities, required_capabilities
from .models import ModelCatalog, is_chat_model
from .ladder import is_local_url, suggest as suggest_ladder
from .pricing import PriceStore, extract_price
from .research import build_prompt as research_prompt, validate as research_validate
from .pool import (
    MODEL,
    NETWORK,
    OVERFLOW,
    REQUEST,
    UNSUPPORTED,
    ProviderPool,
    load_pool_state,
    save_pool_state,
)
from .router import TIERS, TierRouter
from .settings import (
    apply_live,
    build_overlay,
    deep_merge,
    load_overlay,
    merge_config,
    merge_overlay_into_raw,
    merge_raw,
    save_overlay,
    settings_view,
)
from .upstream import (
    build_headers,
    build_payload,
    chat_url,
    classify_upstream_error,
    read_error_detail,
)

CONFIG_ENV_VAR = "INFERENCE_PROXY_CONFIG"
DEFAULT_CONFIG_PATH = "config.yaml"
RECENT_LIMIT = 100

logger = logging.getLogger("inference_proxy")


def create_app(
    config: ProxyConfig | None = None,
    config_path: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    settings_file: str | None = None
    if config is not None:
        raw = config.model_dump()
        settings_file = config.server.settings_file
    else:
        path = config_path or os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG_PATH
        raw = load_config_dict(path)
        settings_file = (raw.get("server") or {}).get("settings_file", "ui-settings.json")

    overlay = load_overlay(settings_file)
    merged = raw
    config = None
    if overlay:
        try:
            merged = merge_raw(raw, overlay)
            config = merge_config(raw, overlay)
        except (ValidationError, ConfigError) as exc:
            print(
                f"warning: settings overlay {settings_file} is invalid and was ignored: {exc}",
                file=sys.stderr,
            )
            merged = raw
    if config is None:
        config = ProxyConfig.model_validate(raw)

    pools: dict[str, ProviderPool] = {
        name: ProviderPool(name, provider) for name, provider in config.providers.items()
    }
    router = TierRouter(pools, config.tiers, config.routing)
    context_registry = ContextRegistry()
    catalog = ModelCatalog()
    prices = PriceStore(config.server.data_db)
    scores = ScoreStore(prices.db)
    judge_state: dict = {"resolved": None, "note": None}
    recent: deque = deque(maxlen=RECENT_LIMIT)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    def hosted_judge(name: str):
        """The judge settings for a provider: its first account, its cheapest-tier model."""
        pool = pools.get(name)
        if pool is None or pool.config.type != "openai" or not pool.accounts:
            return None
        model = pool.config.tier_models.get("trivial") or pool.config.default_model
        if not model:
            return None
        account = pool.accounts[0]
        return config.classifier.model_copy(
            update={"base_url": account.base_url, "api_key": account.api_key, "model": model,
                    "timeout_seconds": max(config.classifier.timeout_seconds, 30.0)}
        )

    def pin_for(base_url: str) -> str | None:
        """The model this endpoint is pinned to, if any."""
        return compute_pins(config).get(endpoint_key(base_url))

    def current_judge():
        """The classifier settings in force: the local server, or a hosted provider chosen by classifier.provider.

        A judge on a pinned endpoint uses the pinned model, so judging never makes that server swap models.
        """
        judge = judge_state["resolved"] or config.classifier
        pinned = pin_for(judge.base_url)
        return judge.model_copy(update={"model": pinned}) if pinned and pinned != judge.model else judge

    async def resolve_judge() -> None:
        wanted = config.classifier.provider
        judge_state["resolved"], judge_state["note"] = None, None
        if not wanted:
            return
        if wanted != "auto":
            judge_state["resolved"] = hosted_judge(wanted)
            judge_state["note"] = f"judge: provider {wanted}" if judge_state["resolved"] else f"judge: provider {wanted} cannot judge (needs an openai-type provider with a model)"
            return
        try:  # any HTTP answer, even an error status, means the local server is there
            await app.state.http.get(config.classifier.base_url.rstrip("/") + "/models", timeout=3.0)
            judge_state["note"] = "judge: local server"
            return
        except httpx.HTTPError:
            pass
        for name in pools:
            candidate = hosted_judge(name)
            if candidate is not None:
                judge_state["resolved"], judge_state["note"] = candidate, f"judge: local server not reachable, using {name}"
                logger.warning(judge_state["note"])
                return
        judge_state["note"] = "judge: nothing reachable, using the length heuristic"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.http = http_client or httpx.AsyncClient()
        app.state.owns_http = http_client is None
        await resolve_judge()
        judge = current_judge()
        # best-effort: learn the judge's context window so oversized prompts
        # can skip the judge instead of triggering overflow loops
        await probe_model_context(
            app.state.http,
            judge.base_url,
            judge.model,
            judge.api_key,
            context_registry,
        )
        if config.server.state_file:
            load_pool_state(config.server.state_file, pools)
        research_task = asyncio.create_task(scheduled_research()) if config.research.enabled else None
        try:
            yield
        finally:
            if research_task:
                research_task.cancel()
            if config.server.state_file:
                try:
                    save_pool_state(config.server.state_file, pools)
                except OSError:
                    pass
            if app.state.owns_http:
                await app.state.http.aclose()

    app = FastAPI(title="inference-proxy", version="0.1.0", lifespan=lifespan)
    app.state.config = config
    app.state.pools = pools
    app.state.router = router
    app.state.catalog = catalog
    app.state.prices = prices
    app.state.scores = scores
    app.state.judge_state = judge_state
    app.state.context_registry = context_registry
    app.state.raw_config = merged
    app.state.settings_file = settings_file

    def check_auth(request: Request) -> JSONResponse | None:
        expected = config.server.api_key
        if not expected:
            return None
        provided = request.headers.get("authorization", "")
        if provided != f"Bearer {expected}":
            return JSONResponse(
                status_code=401,
                content={"error": {"message": "invalid or missing proxy API key"}},
            )
        return None

    def no_capable_provider(required: frozenset[str], tier: str | None = None) -> JSONResponse:
        needed = media_capabilities(required)
        scope = f" in tier '{tier}'" if tier else " across all configured providers"
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": (
                        f"no provider{scope} supports the required capabilities {needed}; "
                        "declare the capability on a provider or pin one explicitly"
                    )
                }
            },
        )

    async def resolve_plan(
        body: dict,
    ) -> tuple[list[tuple[str, ProviderPool]], frozenset[str], int | None, str, str | None] | JSONResponse:
        """Returns (plan, required_capabilities, difficulty, method, model_override) or an error response."""
        requested = str(body.get("model") or "auto").strip()
        required = required_capabilities(body)

        if requested in ("auto", ""):
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                return JSONResponse(
                    status_code=400,
                    content={"error": {"message": "'messages' must be a non-empty list"}},
                )
            difficulty, method = await classify_difficulty(
                app.state.http,
                current_judge(),
                messages,
                context_registry=context_registry,
                difficulty_floor=config.routing.moderate_min,
            )
            tier = router.tier_for(difficulty)
            plan = router.plan(tier, required)
            if not plan:
                return no_capable_provider(required)
            return plan, required, difficulty, method, None

        if requested in TIERS:
            plan = router.plan(requested, required)
            if not plan:
                return no_capable_provider(required, requested)
            return plan, required, None, "forced", None

        pinned_model: str | None = None
        provider_name = requested
        if ":" in requested:
            provider_name, _, pinned_model = requested.partition(":")
        pool = pools.get(provider_name)
        if pool is not None:
            tier = next(
                (t for t, names in config.tiers.items() if provider_name in names),
                "unassigned",
            )
            # pinned means pinned: capability filtering does not apply
            return [(tier, pool)], required, None, "pinned", pinned_model

        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": (
                        f"unknown model '{requested}'; use auto, one of {list(TIERS)}, "
                        "a provider name, or 'provider:model'"
                    )
                }
            },
        )

    async def attempt(
        pool: ProviderPool,
        account,
        payload: dict,
        stream: bool,
    ) -> httpx.Response:
        timeout = httpx.Timeout(
            connect=10.0,
            read=pool.config.timeout_seconds,
            write=30.0,
            pool=30.0,
        )
        if pool.config.type in CLI_TYPES:
            return await run_cli(pool.config.type, payload.get("model"), payload, pool.config.timeout_seconds, stream)
        request = app.state.http.build_request(
            "POST",
            chat_url(account.base_url),
            json=payload,
            headers=build_headers(account.api_key),
            timeout=timeout,
        )
        return await app.state.http.send(request, stream=stream)

    async def chat_completions(request: Request):
        unauthorized = check_auth(request)
        if unauthorized:
            return unauthorized
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse(status_code=400, content={"error": {"message": "invalid JSON body"}})
        if not isinstance(body, dict):
            return JSONResponse(status_code=400, content={"error": {"message": "body must be an object"}})

        started = time.perf_counter()
        requested_model = str(body.get("model") or "auto").strip()
        stream = bool(body.get("stream"))
        resolved = await resolve_plan(body)
        if isinstance(resolved, JSONResponse):
            return resolved
        plan, required, difficulty, difficulty_method, model_override = resolved
        requested_tier = plan[0][0] if plan else None
        caps_beyond_text = media_capabilities(required)

        attempts: list[dict] = []

        def record_request(meta: dict, status: int) -> None:
            entry = {
                "ts": round(time.time(), 3),
                "model": requested_model,
                "difficulty": meta.get("difficulty"),
                "difficulty_method": meta.get("difficulty_method"),
                "tier": meta.get("tier"),
                "provider": meta.get("provider"),
                "account": meta.get("account"),
                "upstream_model": meta.get("model"),
                "capabilities": meta.get("capabilities", []),
                "stream": stream,
                "status": status,
                "ms": round((time.perf_counter() - started) * 1000),
            }
            recent.appendleft(entry)
            logger.info(
                "request model=%s difficulty=%s(%s) tier=%s provider=%s account=%s "
                "upstream_model=%s caps=%s stream=%s status=%s duration=%sms",
                entry["model"], entry["difficulty"], entry["difficulty_method"],
                entry["tier"], entry["provider"], entry["account"],
                entry["upstream_model"], entry["capabilities"], entry["stream"],
                entry["status"], entry["ms"],
            )

        for effective_tier, pool in plan:
            tried_accounts: set[str] = set()
            first_model = model_override or pool.config.tier_models.get(effective_tier) or pool.config.default_model
            candidates: list[str | None] = [first_model]
            model_index = 0
            while True:
                account = pool.acquire()
                if account is None:
                    if not tried_accounts:
                        attempts.append(
                            {
                                "provider": pool.name,
                                "tier": effective_tier,
                                "error": "all accounts cooling down",
                            }
                        )
                    break
                if account.id in tried_accounts:
                    pool.release(account)
                    break
                tried_accounts.add(account.id)

                upstream_model = candidates[model_index]
                pinned_here = None if model_override or pool.config.type in CLI_TYPES else pin_for(account.base_url)
                if pinned_here:  # an explicit provider:model from the client is honoured; the proxy's own choices are not
                    upstream_model = pinned_here
                if not upstream_model and pool.config.type not in CLI_TYPES:
                    pool.release(account)
                    return JSONResponse(
                        status_code=500,
                        content={
                            "error": {
                                "message": f"provider '{pool.name}' has no default_model configured",
                            }
                        },
                    )

                payload = build_payload(
                    body,
                    upstream_model,
                    pool.config,
                    config.tier_policies.get(effective_tier),
                )
                try:
                    response = await attempt(pool, account, payload, stream)
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    pool.report_failure(account, NETWORK)
                    attempts.append(
                        {
                            "provider": pool.name,
                            "tier": effective_tier,
                            "account": account.id,
                            "error": f"network error: {exc}",
                        }
                    )
                    continue

                if response.status_code == 200:
                    pool.report_success(account)
                    meta = {
                        "difficulty": difficulty,
                        "difficulty_method": difficulty_method,
                        "tier": effective_tier,
                        "provider": pool.name,
                        "account": account.id,
                        "model": upstream_model,
                        "capabilities": caps_beyond_text,
                    }
                    record_request(meta, 200)
                    if stream:
                        return stream_response(response, meta)
                    return await json_response(response, meta)

                error_body = await response.aread()
                await response.aclose()
                kind = classify_upstream_error(response.status_code, error_body)
                if kind == MODEL and model_override is None and not caps_beyond_text:
                    # The model is missing or retired, not the account. Try the provider's other
                    # models (never when pinned, and never for media, whose support is per model).
                    pool.release(account)
                    attempts.append(
                        {
                            "provider": pool.name,
                            "tier": effective_tier,
                            "account": account.id,
                            "model": upstream_model,
                            "status": response.status_code,
                            "kind": kind,
                            "error": read_error_detail(response.status_code, error_body),
                        }
                    )
                    if len(candidates) == 1 and not pinned_here:
                        if not pool.config.fallback_models:  # configured stand-ins need no discovery
                            await catalog.ensure(app.state.http, pool)
                        candidates.extend(catalog.siblings(pool, {m for m in candidates if m}))
                    model_index += 1
                    if model_index < len(candidates):
                        tried_accounts.discard(account.id)
                        continue
                    break
                if kind == OVERFLOW:
                    # The request can never fit this provider's context; the
                    # account is healthy, so fail over without a cooldown and
                    # remember the provider's real limit for next time.
                    context_registry.learn_from_error(
                        account.base_url, upstream_model, error_body
                    )
                    pool.release(account)
                    attempts.append(
                        {
                            "provider": pool.name,
                            "tier": effective_tier,
                            "account": account.id,
                            "status": response.status_code,
                            "kind": kind,
                            "error": read_error_detail(response.status_code, error_body),
                        }
                    )
                    continue
                relay = kind == REQUEST or kind == MODEL or (
                    kind == UNSUPPORTED and (effective_tier, pool) == plan[-1]
                )
                if relay:
                    # A request the provider will never accept: relay it. For
                    # unsupported media this only happens when no capable
                    # provider remains to fail over to.
                    pool.release(account)
                    record_request(
                        {
                            "difficulty": difficulty,
                            "difficulty_method": difficulty_method,
                            "tier": effective_tier,
                            "provider": pool.name,
                            "account": account.id,
                            "model": upstream_model,
                            "capabilities": caps_beyond_text,
                        },
                        response.status_code,
                    )
                    return relay_upstream_error(response.status_code, error_body, pool.name)
                pool.report_failure(account, kind)
                attempts.append(
                    {
                        "provider": pool.name,
                        "tier": effective_tier,
                        "account": account.id,
                        "status": response.status_code,
                        "kind": kind,
                        "error": read_error_detail(response.status_code, error_body),
                    }
                )

        record_request(
            {
                "difficulty": difficulty,
                "difficulty_method": difficulty_method,
                "tier": requested_tier,
                "provider": None,
                "account": None,
                "model": None,
                "capabilities": caps_beyond_text,
            },
            502,
        )
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": "all upstream providers failed",
                    "attempts": attempts,
                }
            },
        )

    def meta_headers(meta: dict) -> dict[str, str]:
        headers = {
            "X-Proxy-Tier": str(meta["tier"]),
            "X-Proxy-Provider": str(meta["provider"]),
            "X-Proxy-Account": str(meta["account"]),
        }
        if meta["difficulty"] is not None:
            headers["X-Proxy-Difficulty"] = str(meta["difficulty"])
            headers["X-Proxy-Difficulty-Method"] = str(meta["difficulty_method"])
        return headers

    def stream_response(response: httpx.Response, meta: dict) -> StreamingResponse:
        async def relay():
            # SSE comment: ignored by spec-compliant clients, visible when
            # debugging, and carries the routing decision for streams.
            yield f": {json.dumps({'inference_proxy': meta})}\n\n".encode()
            seen_usage: dict | None = None
            pending = b""
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
                    if b'"usage"' in chunk or pending:  # cheap check: most chunks carry no usage
                        pending = (pending + chunk)[-8192:]
                        for line in pending.split(b"\n"):
                            if line.startswith(b"data:") and b'"usage"' in line:
                                try:
                                    found = json.loads(line[5:]).get("usage")
                                except (ValueError, AttributeError):
                                    continue
                                if isinstance(found, dict):
                                    seen_usage = found
                        pending = pending.rsplit(b"\n", 1)[-1] if b"\n" in pending else pending
            except httpx.HTTPError:
                payload = json.dumps({"error": {"message": "upstream connection lost mid-stream"}})
                yield f"\n\ndata: {payload}\n\n".encode()
            finally:
                await response.aclose()
                prices.record_usage(meta["provider"], meta["model"], meta["tier"], seen_usage)

        return StreamingResponse(
            relay(),
            status_code=200,
            media_type=response.headers.get("content-type", "text/event-stream"),
            headers=meta_headers(meta),
            background=BackgroundTask(response.aclose),
        )

    async def json_response(response: httpx.Response, meta: dict) -> JSONResponse:
        try:
            await response.aread()
            data = response.json()
        except ValueError:
            await response.aclose()
            return JSONResponse(
                status_code=502,
                content={"error": {"message": "upstream returned invalid JSON"}},
            )
        await response.aclose()
        if isinstance(data, dict):
            data["inference_proxy"] = meta
            prices.record_usage(meta["provider"], meta["model"], meta["tier"], data.get("usage"))
        return JSONResponse(status_code=200, content=data, headers=meta_headers(meta))

    def relay_upstream_error(status: int, error_body: bytes, provider: str) -> JSONResponse:
        return JSONResponse(
            status_code=status,
            content={
                "error": {
                    "message": read_error_detail(status, error_body),
                    "provider": provider,
                }
            },
        )

    async def models(request: Request):
        unauthorized = check_auth(request)
        if unauthorized:
            return unauthorized
        data = [{"id": "auto", "object": "model"}]
        data += [{"id": tier, "object": "model"} for tier in TIERS]
        for name, provider in config.providers.items():
            data.append({"id": name, "object": "model"})
            if provider.default_model:
                data.append({"id": f"{name}:{provider.default_model}", "object": "model"})
        return JSONResponse(content={"object": "list", "data": data})

    async def pools_status(request: Request):
        unauthorized = check_auth(request)
        if unauthorized:
            return unauthorized
        return JSONResponse(content={"providers": [pool.snapshot() for pool in pools.values()]})

    async def get_settings(request: Request):
        unauthorized = check_auth(request)
        if unauthorized:
            return unauthorized
        return JSONResponse(content=settings_view(config, pools, app.state.settings_file))

    async def update_settings(request: Request):
        unauthorized = check_auth(request)
        if unauthorized:
            return unauthorized
        try:
            payload = await request.json()
        except ValueError:
            return JSONResponse(status_code=400, content={"error": {"message": "invalid JSON body"}})
        if not isinstance(payload, dict):
            return JSONResponse(status_code=400, content={"error": {"message": "body must be an object"}})

        overlay = build_overlay(payload, pools)
        settings_file = app.state.settings_file
        combined = deep_merge(load_overlay(settings_file), overlay)
        new_merged = merge_overlay_into_raw(app.state.raw_config, combined)
        try:
            new_config = ProxyConfig.model_validate(new_merged)
        except (ValidationError, ConfigError, ValueError) as exc:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": f"invalid settings: {exc}"}},
            )

        apply_live(config, pools, router, new_config)
        app.state.raw_config = new_merged
        if settings_file:
            try:
                save_overlay(settings_file, combined)
            except OSError as exc:
                return JSONResponse(
                    status_code=500,
                    content={"error": {"message": f"failed to save settings: {exc}"}},
                )
        return JSONResponse(content=settings_view(config, pools, app.state.settings_file))

    async def reset_account(request: Request, provider: str, index: int):
        unauthorized = check_auth(request)
        if unauthorized:
            return unauthorized
        pool = pools.get(provider)
        if pool is None or pool.reset_account(index) is None:
            return JSONResponse(
                status_code=404,
                content={"error": {"message": f"no account {provider}#{index}"}},
            )
        return JSONResponse(content={"ok": True, "providers": [p.snapshot() for p in pools.values()]})

    async def recent_requests(request: Request):
        unauthorized = check_auth(request)
        if unauthorized:
            return unauthorized
        return JSONResponse(content={"requests": list(recent)})

    async def ui_page(request: Request):
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    async def models_catalog(request: Request):
        unauthorized = check_auth(request)
        if unauthorized:
            return unauthorized
        return JSONResponse(content={"providers": catalog.snapshot(), "judge": judge_state["note"], "pins": compute_pins(config)})

    async def models_refresh(request: Request):
        unauthorized = check_auth(request)
        if unauthorized:
            return unauthorized
        for pool in pools.values():
            await catalog.refresh(app.state.http, pool)
        return JSONResponse(content={"providers": catalog.snapshot(), "judge": judge_state["note"], "pins": compute_pins(config)})

    def price_view() -> dict:
        now = time.time()
        rows = prices.listing()
        for row in rows:
            row["age_days"] = round((now - row["fetched_at"]) / 86400, 1) if row.get("fetched_at") else None
            row["stale"] = bool(row["age_days"] is not None and row["age_days"] > config.server.price_stale_days)
        return {"prices": rows, "stale_after_days": config.server.price_stale_days}

    async def refresh_prices(everything: bool = False) -> list[dict]:
        """Ask each provider for its models and prices, and file the differences as proposals."""
        changes: list[dict] = []
        for name, pool in pools.items():
            if pool.config.type != "openai":
                continue
            entry = await catalog.refresh(app.state.http, pool)
            if entry["error"] or not entry["raw"]:
                continue
            wanted = set(entry["raw"]) if everything else {
                m for m in (pool.config.default_model, *pool.config.tier_models.values(), *pool.config.fallback_models) if m
            } | {r["model"] for r in prices.listing("active") if r["provider"] == name}
            for model in sorted(wanted):
                if model in entry["raw"]:
                    price = extract_price(entry["raw"][model])
                    if price is None:
                        continue
                else:
                    price = None  # reported earlier, gone now
                kind = prices.propose(name, model, price, "provider /models", pool.accounts[0].base_url.rstrip("/") + "/models" if pool.accounts else None)
                if kind:
                    changes.append({"provider": name, "model": model, "kind": kind})
        return changes

    async def refresh_scores(source: str = "all") -> dict:
        """Download independent scores. Epoch needs nothing; Artificial Analysis needs your own free key."""
        done: dict = {}
        if source in ("epoch", "all"):
            try:
                done["epoch"] = scores.replace("epoch", await fetch_epoch(app.state.http))
            except (httpx.HTTPError, ValueError, OSError) as exc:
                done["epoch"] = f"failed: {exc}"[:200]
        if source in ("aa", "all"):
            key = config.benchmarks.aa_api_key
            if not key:
                done["aa"] = "skipped: no Artificial Analysis API key (benchmarks.aa_api_key)"
            else:
                try:
                    done["aa"] = scores.replace("aa", await fetch_aa(app.state.http, key))
                except (httpx.HTTPError, ValueError) as exc:
                    done["aa"] = f"failed: {exc}"[:200]
        return done

    def ladder_candidates(source: str | None) -> list[dict]:
        """Models that could fill a rung: one entry per endpoint and model, pinned endpoints limited to their one model."""
        seen, out = set(), []
        for name, pool in pools.items():
            if pool.config.type != "openai" or not pool.accounts:
                continue
            base = pool.accounts[0].base_url
            pinned = pin_for(base)
            entry = catalog.entries.get(name) or {}
            models = [pinned] if pinned else [m for m in (entry.get("models") or [pool.config.default_model]) if m]
            for model in models:
                key = (endpoint_key(base), model)
                if key in seen:
                    continue
                seen.add(key)
                found = scores.lookup(model, source)
                price = prices.active(name, model)
                out.append({"provider": name, "model": model, "score": found["score"] if found else None,
                            "tokens_per_s": (found or {}).get("detail", {}).get("tokens_per_s"),
                            "score_source": found["source"] if found else None,
                            "input_per_m": price["input_per_m"] if price else None, "output_per_m": price["output_per_m"] if price else None,
                            "pinned": bool(pinned), "local": is_local_url(base)})
        return [c for c in out if is_chat_model(c["model"])]

    async def scores_get(request: Request):
        unauthorized = check_auth(request)
        return unauthorized or JSONResponse(content={"sources": scores.summary()})

    async def scores_refresh(request: Request):
        unauthorized = check_auth(request)
        if unauthorized:
            return unauthorized
        try:
            body = await request.json()
        except ValueError:
            body = {}
        source = (body or {}).get("source", "all")
        if source not in ("epoch", "aa", "all"):
            return JSONResponse(status_code=400, content={"error": {"message": "source must be epoch, aa or all"}})
        return JSONResponse(content={"result": await refresh_scores(source), "sources": scores.summary()})

    async def ladder_get(request: Request):
        unauthorized = check_auth(request)
        if unauthorized:
            return unauthorized
        try:
            rungs = int(request.query_params.get("rungs", 10))
        except ValueError:
            rungs = 10
        try:
            tolerance = max(0.0, min(0.2, float(request.query_params.get("tolerance", 0.03))))
        except ValueError:
            tolerance = 0.03
        source = request.query_params.get("source") or None
        if source not in (None, *SOURCES):
            return JSONResponse(status_code=400, content={"error": {"message": f"source must be one of {list(SOURCES)}"}})
        result = suggest_ladder(ladder_candidates(source), rungs, tolerance)
        patch = {}
        for tier, pick in result["tiers"].items():
            patch.setdefault(pick["provider"], {"tier_models": {}})["tier_models"][tier] = pick["model"]
        return JSONResponse(content={**result, "suggested_tier_models": patch, "sources": scores.summary()})

    async def research_prices(provider: str, cli_provider: str, url: str | None = None) -> dict:
        """Ask a Claude CLI provider to read a pricing page; file what it finds as proposals only."""
        pool, cli_pool = pools.get(provider), pools.get(cli_provider)
        if pool is None or cli_pool is None or cli_pool.config.type != "claude":
            raise ValueError("research needs a known provider and a provider of type claude")
        entry = await catalog.ensure(app.state.http, pool)
        known = [m for m in entry["models"] if is_chat_model(m)]
        text = await ask_cli("claude", research_prompt(known, url), cli_pool.config.default_model, cli_pool.config.timeout_seconds, web=True)
        match = re.search(r"\{.*\}", text, re.S)
        if match is None:
            raise ValueError("the research answer was not JSON, so nothing was read")
        rows = research_validate(json.loads(match.group(0)), known)
        changes = []
        for row in rows:
            kind = prices.propose(provider, row["model"], (row["input_per_m"], row["output_per_m"], None), "web research", row["source_url"])
            if kind:
                changes.append({"provider": provider, "model": row["model"], "kind": kind})
        return {"read": len(rows), "changes": changes}

    async def research_post(request: Request):
        unauthorized = check_auth(request)
        if unauthorized:
            return unauthorized
        body = await request.json()
        try:
            cli = body.get("cli_provider") or config.research.cli_provider
            provider = str(body["provider"])
            result = await research_prices(provider, str(cli), body.get("url") or config.research.price_pages.get(provider))
        except (KeyError, ValueError, RuntimeError) as exc:
            return JSONResponse(status_code=400, content={"error": {"message": str(exc)[:300]}})
        return JSONResponse(content={**result, **price_view()})

    async def scheduled_research() -> None:
        """If the owner turned research on: refresh prices and scores, and read price pages, every few days. Proposals only."""
        while True:
            last = float(prices.meta_get("research_last_run") or 0)
            wait = last + config.research.every_days * 86400 - time.time()
            if wait > 0:
                await asyncio.sleep(min(wait, 3600))
                continue
            try:
                await refresh_prices(False)
                await refresh_scores("all")
                for provider, url in config.research.price_pages.items():
                    if config.research.cli_provider and provider in pools:
                        try:
                            await research_prices(provider, config.research.cli_provider, url)
                        except (ValueError, RuntimeError) as exc:
                            logger.warning("research: %s: %s", provider, exc)
                logger.info("research: scheduled refresh finished; proposals are waiting in the admin page")
            except Exception as exc:  # a failed pass must not end the schedule
                logger.warning("research: scheduled refresh failed: %s", exc)
            prices.meta_set("research_last_run", str(time.time()))

    async def prices_get(request: Request):
        unauthorized = check_auth(request)
        return unauthorized or JSONResponse(content=price_view())

    async def prices_refresh(request: Request):
        unauthorized = check_auth(request)
        if unauthorized:
            return unauthorized
        try:
            body = await request.json()
        except ValueError:
            body = {}
        changes = await refresh_prices(bool((body or {}).get("all")))
        return JSONResponse(content={"changes": changes, **price_view()})

    async def prices_apply(request: Request):
        unauthorized = check_auth(request)
        if unauthorized:
            return unauthorized
        body = await request.json()
        return JSONResponse(content={"applied": prices.apply(body.get("ids") or None), **price_view()})

    async def prices_reject(request: Request):
        unauthorized = check_auth(request)
        if unauthorized:
            return unauthorized
        body = await request.json()
        return JSONResponse(content={"rejected": prices.reject(body.get("ids") or None), **price_view()})

    async def prices_set(request: Request):
        unauthorized = check_auth(request)
        if unauthorized:
            return unauthorized
        body = await request.json()
        try:
            prices.set_manual(str(body["provider"]), str(body["model"]), float(body["input_per_m"]), float(body["output_per_m"]), body.get("context"))
        except (KeyError, TypeError, ValueError) as exc:
            return JSONResponse(status_code=400, content={"error": {"message": f"bad price: {exc}"}})
        return JSONResponse(content=price_view())

    async def usage_get(request: Request):
        unauthorized = check_auth(request)
        if unauthorized:
            return unauthorized
        try:
            days = float(request.query_params.get("days", 7))
        except ValueError:
            days = 7.0
        rows = prices.usage_summary(days)
        return JSONResponse(content={"days": days, "usage": rows, "total_cost_usd": round(sum(r["cost_usd"] or 0 for r in rows), 6)})

    async def healthz(request: Request):
        return JSONResponse(content={"ok": True})

    app.add_api_route("/v1/chat/completions", chat_completions, methods=["POST"])
    app.add_api_route("/v1/models", models, methods=["GET"])
    app.add_api_route("/admin/pools", pools_status, methods=["GET"])
    app.add_api_route("/admin/api/settings", get_settings, methods=["GET"])
    app.add_api_route("/admin/api/settings", update_settings, methods=["PUT"])
    app.add_api_route("/admin/api/pools/{provider}/accounts/{index}/reset", reset_account, methods=["POST"])
    app.add_api_route("/admin/api/recent", recent_requests, methods=["GET"])
    app.add_api_route("/", ui_page, methods=["GET"])
    app.add_api_route("/admin", ui_page, methods=["GET"])
    app.add_api_route("/admin/api/models", models_catalog, methods=["GET"])
    app.add_api_route("/admin/api/models/refresh", models_refresh, methods=["POST"])
    app.add_api_route("/admin/api/prices", prices_get, methods=["GET"])
    app.add_api_route("/admin/api/prices", prices_set, methods=["PUT"])
    app.add_api_route("/admin/api/prices/refresh", prices_refresh, methods=["POST"])
    app.add_api_route("/admin/api/prices/apply", prices_apply, methods=["POST"])
    app.add_api_route("/admin/api/prices/reject", prices_reject, methods=["POST"])
    app.add_api_route("/admin/api/usage", usage_get, methods=["GET"])
    app.add_api_route("/admin/api/scores", scores_get, methods=["GET"])
    app.add_api_route("/admin/api/scores/refresh", scores_refresh, methods=["POST"])
    app.add_api_route("/admin/api/ladder", ladder_get, methods=["GET"])
    app.add_api_route("/admin/api/research/prices", research_post, methods=["POST"])
    app.add_api_route("/healthz", healthz, methods=["GET"])
    return app


def run() -> None:
    try:
        config = load_config(os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG_PATH)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "hint: copy config.example.yaml to config.yaml and fill in your keys,"
            " or set INFERENCE_PROXY_CONFIG to a config file path.",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
    uvicorn.run(
        create_app(config),
        host=config.server.host,
        port=config.server.port,
    )
