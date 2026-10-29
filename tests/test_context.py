import httpx

from inference_proxy.classifier import oversize_limit_chars
from inference_proxy.config import ClassifierConfig
from inference_proxy.context import (
    CHARS_PER_TOKEN,
    ContextRegistry,
    find_context_tokens,
    looks_like_context_overflow,
    parse_overflow_tokens,
    probe_model_context,
)


def test_parse_overflow_tokens_known_message_shapes():
    openai = "This model's maximum context length is 8192 tokens. However, you requested 10048 tokens (7000 in the messages, 3048 in the completion)."
    assert parse_overflow_tokens(openai) == 8192
    llamacpp = "Requested tokens (9000) exceed context window (8192)."
    assert parse_overflow_tokens(llamacpp) == 8192
    anthropic = "prompt is too long: 250000 tokens > 200000 maximum"
    assert parse_overflow_tokens(anthropic) == 200000
    assert parse_overflow_tokens("the request exceeds the available context size") is None
    assert parse_overflow_tokens("max_tokens must be positive") is None


def test_looks_like_context_overflow_markers():
    assert looks_like_context_overflow("This model's maximum context length is 8192 tokens")
    assert looks_like_context_overflow("Requested tokens exceed context window")
    assert looks_like_context_overflow("Prompt is too long: 300000 tokens > 200000 maximum")
    assert not looks_like_context_overflow("max_tokens must be positive")
    assert not looks_like_context_overflow("You exceeded your current quota")


def test_find_context_tokens_prefers_loaded_instance_config():
    payload = {
        "models": [
            {
                "key": "qwen/qwen3.5",
                "max_context_length": 262144,
                "loaded_instances": [
                    {"id": "qwen/qwen3.5", "config": {"context_length": 8192}}
                ],
            }
        ]
    }
    assert find_context_tokens(payload, "qwen3.5") == 8192


def test_find_context_tokens_falls_back_to_model_maximum():
    payload = {"models": [{"key": "deepseek-r1", "max_context_length": 131072, "loaded_instances": []}]}
    assert find_context_tokens(payload, "deepseek-r1") == 131072


def test_find_context_tokens_matches_instance_id_and_suffix():
    payload = {
        "models": [
            {
                "key": "vendor/model-x",
                "max_context_length": 32768,
                "loaded_instances": [{"id": "rater", "config": {"context_length": 4096}}],
            }
        ]
    }
    assert find_context_tokens(payload, "rater") == 4096
    assert find_context_tokens(payload, "model-x") == 4096
    assert find_context_tokens(payload, "other") is None
    assert find_context_tokens({"object": "list", "data": []}, "rater") is None


async def test_probe_uses_lmstudio_v1_and_learns():
    seen = []
    payload = {
        "models": [
            {
                "key": "qwen/qwen3.5",
                "max_context_length": 262144,
                "loaded_instances": [{"id": "qwen/qwen3.5", "config": {"context_length": 8192}}],
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.path == "/api/v1/models":
            return httpx.Response(200, json=payload)
        return httpx.Response(404, json={"error": {"message": "not found"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    registry = ContextRegistry()
    try:
        tokens = await probe_model_context(
            client, "http://judge.test/v1", "qwen3.5", registry=registry
        )
    finally:
        await client.aclose()
    assert tokens == 8192
    assert seen == ["http://judge.test/api/v1/models"]
    assert registry.known_tokens("http://judge.test/v1", "qwen3.5") == 8192


async def test_probe_falls_back_to_v0_endpoint():
    seen = []
    payload = {"data": [{"key": "rater", "max_context_length": 4096, "loaded_instances": []}]}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/api/v0/models":
            return httpx.Response(200, json=payload)
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        tokens = await probe_model_context(client, "http://judge.test/v1", "rater")
    finally:
        await client.aclose()
    assert tokens == 4096
    assert seen == ["/api/v1/models", "/api/v0/models"]


async def test_probe_returns_none_without_lmstudio_api():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        tokens = await probe_model_context(client, "http://api.test/v1", "gpt-x")
    finally:
        await client.aclose()
    assert tokens is None


def test_registry_learn_and_ignore_invalid():
    registry = ContextRegistry()
    registry.learn_tokens("http://x.test/v1", "m", 8192)
    registry.learn_tokens("http://x.test/v1", "m", 0)
    registry.learn_tokens("http://x.test/v1", "m", -4)
    assert registry.known_tokens("http://x.test/v1", "m") == 8192
    assert registry.known_tokens("http://x.test/v1", "other") is None


def test_registry_learns_from_overflow_error_body():
    registry = ContextRegistry()
    body = b'{"error": {"message": "Requested tokens (9000) exceed context window (8192)."}}'
    assert registry.learn_from_error("http://x.test/v1", "m", body) is True
    assert registry.known_tokens("http://x.test/v1", "m") == 8192
    assert registry.learn_from_error("http://x.test/v1", "m", b'{"error": "quota"}') is False


def test_oversize_limit_from_registry_and_override():
    config = ClassifierConfig(base_url="http://rater.test/v1", model="rater")
    assert oversize_limit_chars(config, None) is None
    registry = ContextRegistry()
    registry.learn_tokens("http://rater.test/v1", "rater", 8192)
    # 8192 tokens * 3 chars/token * 0.5 ratio
    assert oversize_limit_chars(config, registry) == int(8192 * CHARS_PER_TOKEN * 0.5)
    override = ClassifierConfig(
        base_url="http://rater.test/v1", model="rater", context_chars=6000, oversize_ratio=0.5
    )
    assert oversize_limit_chars(override, registry) == 3000
