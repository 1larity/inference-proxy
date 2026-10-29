# Commands
- Test: `uv run pytest`
- Run: `uv run inference-proxy` (config: ./config.yaml or $INFERENCE_PROXY_CONFIG)
- Lint/typecheck: none configured yet

# Layout
- src/inference_proxy/: config.py (YAML+env), pool.py (account rotation/cooldowns), classifier.py (difficulty judge + heuristic fallback), router.py (tiers), upstream.py (error classification), settings.py (UI settings overlay), main.py (FastAPI app + admin API), static/index.html (admin UI)
- tests/: pytest + pytest-asyncio, upstreams mocked via httpx.MockTransport (see tests/conftest.py)

# Notes
- ui-settings.json is the UI settings overlay (contains plaintext keys — never commit)
- Settings hot-apply at runtime; tests default settings_file=None to avoid stray files
- Providers are generic: any name/count; overlay deletion marker is "providers": {"name": null}; tiers are fixed easy/moderate/hard
