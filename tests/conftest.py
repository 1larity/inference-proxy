"""Shared fixtures: config builder and a scriptable MockTransport upstream."""

from __future__ import annotations

import json

import httpx
import pytest

from inference_proxy.config import (
    AccountConfig,
    ClassifierConfig,
    ProviderConfig,
    ProxyConfig,
    RoutingConfig,
    ServerConfig,
)
from inference_proxy.main import create_app


def completion_json(text: str, model: str = "test") -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def make_config(**overrides) -> ProxyConfig:
    server = overrides.pop("server", None) or ServerConfig(settings_file=None)
    if server.settings_file == "ui-settings.json":
        # never let tests read a real overlay dropped in the repo root
        server = server.model_copy(update={"settings_file": None})
    defaults = dict(
        server=server,
        classifier=ClassifierConfig(base_url="http://classifier.test/v1", model="rater"),
        routing=RoutingConfig(moderate_min=4, hard_min=7),
        providers={
            "local": ProviderConfig(
                base_url="http://local.test/v1",
                default_model="qwen3.5",
                capabilities=["text"],
            ),
            "zai": ProviderConfig(
                base_url="http://zai.test/v1",
                default_model="glm-4.7",
                capabilities=["text", "image", "video", "reasoning"],
                accounts=[
                    AccountConfig(api_key="zai-key-1"),
                    AccountConfig(api_key="zai-key-2"),
                ],
            ),
            "openrouter": ProviderConfig(
                base_url="http://or.test/v1",
                default_model="deepseek/deepseek-chat-v3.1",
                api_key="or-key-1",
                capabilities=["text"],
            ),
            "kilo": ProviderConfig(
                base_url="http://kilo.test/v1",
                default_model="kilo-model",
                api_key="kilo-key-1",
                capabilities=["text", "image", "reasoning"],
            ),
        },
        tiers={
            "easy": ["local"],
            "moderate": ["zai"],
            "hard": ["openrouter", "kilo"],
        },
    )
    defaults.update(overrides)
    return ProxyConfig(**defaults)


class SSEStream(httpx.AsyncByteStream):
    def __init__(self, *chunks: bytes) -> None:
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


class FakeUpstreams:
    """httpx MockTransport handler scripted per test."""

    def __init__(self) -> None:
        self.classifier_score = "5"
        self.classifier_calls = 0
        self.zai_status_by_key = {"zai-key-1": 429, "zai-key-2": 200}
        self.local_status = 200
        self.openrouter_status = 200
        self.kilo_status = 200
        self.media_reject_hosts: set[str] = set()
        self.overflow_hosts: set[str] = set()
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        host = request.url.host
        auth = request.headers.get("Authorization", "")
        key = auth.removeprefix("Bearer ")

        if host == "classifier.test":
            if request.method == "GET":
                # context probe: this server exposes no LM Studio REST API
                return httpx.Response(404, json={"error": {"message": "not found"}})
            self.classifier_calls += 1
            return httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": self.classifier_score}}]},
            )
        if host in self.overflow_hosts:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Requested tokens (9000) exceed context window (8192)."
                    }
                },
            )
        if host in self.media_reject_hosts:
            return httpx.Response(
                400,
                json={"error": {"message": "this model does not support image content"}},
            )
        if host == "local.test":
            return self._chat(self.local_status, "local-reply", request)
        if host == "zai.test":
            status = self.zai_status_by_key.get(key, 200)
            return self._chat(status, f"zai-reply-via-{key}", request)
        if host == "or.test":
            return self._chat(self.openrouter_status, "or-reply", request)
        if host == "kilo.test":
            return self._chat(self.kilo_status, "kilo-reply", request)
        return httpx.Response(404, json={"error": {"message": "unknown host"}})

    def _chat(self, status: int, text: str, request: httpx.Request) -> httpx.Response:
        if status != 200:
            return httpx.Response(status, json={"error": {"message": f"upstream failure {status}"}})
        body = json.loads(request.content)
        return httpx.Response(200, json=completion_json(text, model=body.get("model", "?")))

    def sse_response(self) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=SSEStream(
                b'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n',
                b"data: [DONE]\n\n",
            ),
        )


@pytest.fixture
async def proxy():
    config = make_config()
    upstreams = FakeUpstreams()
    app = create_app(config, http_client=httpx.AsyncClient(transport=httpx.MockTransport(upstreams)))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            yield client, app, upstreams


def chat_body(**overrides) -> dict:
    body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Hello there"}],
    }
    body.update(overrides)
    return body
