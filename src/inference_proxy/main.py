"""FastAPI application exposing an OpenAI-compatible, difficulty-routing proxy."""

from __future__ import annotations

import json
import logging
import os
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
from .config import ConfigError, ProxyConfig, load_config, load_config_dict
from .context import ContextRegistry, probe_model_context
from .media import media_capabilities, required_capabilities
from .pool import (
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
    recent: deque = deque(maxlen=RECENT_LIMIT)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.http = http_client or httpx.AsyncClient()
        app.state.owns_http = http_client is None
        # best-effort: learn the judge's context window so oversized prompts
        # can skip the judge instead of triggering overflow loops
        await probe_model_context(
            app.state.http,
            config.classifier.base_url,
            config.classifier.model,
            config.classifier.api_key,
            context_registry,
        )
        if config.server.state_file:
            load_pool_state(config.server.state_file, pools)
        try:
            yield
        finally:
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
                config.classifier,
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

                upstream_model = model_override or pool.config.default_model
                if not upstream_model:
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
                relay = kind == REQUEST or (
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
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            except httpx.HTTPError:
                payload = json.dumps({"error": {"message": "upstream connection lost mid-stream"}})
                yield f"\n\ndata: {payload}\n\n".encode()
            finally:
                await response.aclose()

        return StreamingResponse(
            relay(),
            status_code=200,
            media_type=response.headers.get("content-type", "text/event-stream"),
            headers=meta_headers(meta),
            background=BackgroundTask(response.aclose),
        )

    async def json_response(response: httpx.Response, meta: dict) -> JSONResponse:
        try:
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
