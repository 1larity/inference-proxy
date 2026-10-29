import httpx

from inference_proxy.classifier import (
    build_transcript,
    classify_difficulty,
    extract_score,
    heuristic_score,
    message_text,
)
from inference_proxy.config import ClassifierConfig
from inference_proxy.context import ContextRegistry


def test_extract_score():
    assert extract_score("7") == 7
    assert extract_score("Difficulty: 10") == 10
    assert extract_score("I'd rate this 3/10.") == 3
    assert extract_score("banana") is None
    assert extract_score("twelve") is None


def test_message_text_handles_parts():
    assert message_text("plain") == "plain"
    parts = [
        {"type": "text", "text": "hello "},
        {"type": "image_url", "image_url": {"url": "x"}},
        {"type": "text", "text": "world"},
    ]
    assert message_text(parts) == "hello \n[image]\nworld"


def test_message_text_marks_all_media_kinds():
    parts = [
        {"type": "video_url", "video_url": "x"},
        {"type": "input_audio", "input_audio": {"data": "x"}},
        {"type": "file", "file": {"file_id": "x"}},
    ]
    assert message_text(parts) == "[video]\n[audio]\n[file]"


def test_build_transcript():
    messages = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "write tests"},
    ]
    assert build_transcript(messages) == "System: be brief\nUser: write tests"


def test_heuristic_simple_vs_hard():
    assert heuristic_score("hi") <= 3
    hard = (
        "Prove the time complexity of this distributed algorithm, "
        "derive the concurrency invariants, and optimize the architecture. "
        "The proof needs to be formal and cover race conditions across "
        "multiple nodes with cryptographic guarantees. " * 4
    )
    assert heuristic_score(hard) >= 7


def test_heuristic_media_floor():
    # media needs a vision-capable model even when the request is trivial
    assert heuristic_score("[image]\nhi") >= 4
    assert heuristic_score("describe [video] and [audio]") >= 4
    assert heuristic_score("no media here") < 4


async def test_classify_uses_local_llm():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": " 8 "}}]},
        )

    config = ClassifierConfig(base_url="http://rater.test/v1", model="rater")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        score, method = await classify_difficulty(
            client, config, [{"role": "user", "content": "do a thing"}]
        )
    finally:
        await client.aclose()
    assert (score, method) == (8, "llm")
    assert captured["url"] == "http://rater.test/v1/chat/completions"
    assert captured["payload"]["model"] == "rater"
    assert captured["payload"]["max_tokens"] == 8


async def test_classify_falls_back_to_heuristic_on_garbage():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "banana"}}]},
        )

    config = ClassifierConfig(base_url="http://rater.test/v1")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        score, method = await classify_difficulty(client, config, [{"role": "user", "content": "hi"}])
    finally:
        await client.aclose()
    assert method == "heuristic"
    assert 1 <= score <= 10


async def test_classify_falls_back_on_connection_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    config = ClassifierConfig(base_url="http://rater.test/v1")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        score, method = await classify_difficulty(
            client, config, [{"role": "user", "content": "hello"}]
        )
    finally:
        await client.aclose()
    assert method == "heuristic"


async def test_oversize_prompt_skips_judge_and_applies_floor():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "1"}}]}
        )

    config = ClassifierConfig(base_url="http://rater.test/v1", model="rater", context_chars=600)
    messages = [{"role": "user", "content": "x" * 350}]  # over the 300-char limit
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        score, method = await classify_difficulty(
            client, config, messages, difficulty_floor=4
        )
    finally:
        await client.aclose()
    assert calls["n"] == 0
    assert method == "oversize"
    # the heuristic alone would score 2; the floor lifts it into moderate
    assert score == 4


async def test_prompt_under_oversize_threshold_still_judged():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "7"}}]}
        )

    config = ClassifierConfig(base_url="http://rater.test/v1", model="rater", context_chars=600)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        score, method = await classify_difficulty(
            client, config, [{"role": "user", "content": "x" * 200}]
        )
    finally:
        await client.aclose()
    assert calls["n"] == 1
    assert (score, method) == (7, "llm")


async def test_oversize_guard_off_without_context_knowledge():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "3"}}]}
        )

    config = ClassifierConfig(base_url="http://rater.test/v1", model="rater")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        score, method = await classify_difficulty(
            client, config, [{"role": "user", "content": "x" * 50000}]
        )
    finally:
        await client.aclose()
    assert calls["n"] == 1
    assert (score, method) == (3, "llm")


async def test_judge_overflow_error_learns_context():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "Requested tokens (9000) exceed context window (8192)."
                }
            },
        )

    registry = ContextRegistry()
    config = ClassifierConfig(base_url="http://rater.test/v1", model="rater")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        score, method = await classify_difficulty(
            client, config, [{"role": "user", "content": "hello"}], context_registry=registry
        )
    finally:
        await client.aclose()
    assert method == "heuristic"
    assert registry.known_tokens("http://rater.test/v1", "rater") == 8192
