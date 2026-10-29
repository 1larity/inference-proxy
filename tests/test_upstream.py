from inference_proxy.config import ProviderConfig, TierPolicy
from inference_proxy.pool import OVERFLOW, QUOTA, REQUEST, UNSUPPORTED
from inference_proxy.upstream import build_payload, classify_upstream_error, looks_like_media_rejection


def test_client_explicit_beats_tier_and_provider():
    provider = ProviderConfig(
        base_url="http://x.test/v1",
        extra_body={"thinking": {"type": "enabled"}},
    )
    tier = TierPolicy(max_tokens=512, extra_body={"thinking": {"type": "disabled"}})
    payload = build_payload(
        {
            "model": "m",
            "messages": [],
            "max_tokens": 16000,
            "thinking": {"type": "enabled", "effort": "high"},
        },
        "glm",
        provider,
        tier,
    )
    assert payload["thinking"] == {"type": "enabled", "effort": "high"}
    assert payload["max_tokens"] == 16000


def test_tier_policy_fills_gaps_and_beats_provider_defaults():
    provider = ProviderConfig(
        base_url="http://x.test/v1",
        extra_body={"thinking": {"type": "enabled"}, "top_p": 0.9},
    )
    tier = TierPolicy(max_tokens=2048, extra_body={"thinking": {"type": "disabled"}})
    payload = build_payload({"model": "m", "messages": []}, "glm", provider, tier)
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["top_p"] == 0.9
    assert payload["max_tokens"] == 2048


def test_provider_cap_is_absolute_regardless_of_client():
    provider = ProviderConfig(base_url="http://x.test/v1", max_tokens_cap=4096)
    tier = TierPolicy(max_tokens=2048)
    payload = build_payload({"model": "m", "messages": [], "max_tokens": 16000}, "m", provider, tier)
    assert payload["max_tokens"] == 4096


def test_tier_default_max_tokens_not_applied_over_client():
    tier = TierPolicy(max_tokens=2048)
    payload = build_payload({"model": "m", "messages": [], "max_tokens": 999}, "m", None, tier)
    assert payload["max_tokens"] == 999


def test_no_policy_leaves_body_untouched():
    payload = build_payload({"model": "m", "max_tokens": 5}, "m")
    assert payload["max_tokens"] == 5


def test_media_rejection_classified_unsupported():
    body = b'{"error": {"message": "this model does not support image content"}}'
    assert classify_upstream_error(400, body) == UNSUPPORTED


def test_context_overflow_classified_overflow_not_quota():
    llamacpp = b'{"error": {"message": "Requested tokens (9000) exceed context window (8192)."}}'
    assert classify_upstream_error(400, llamacpp) == OVERFLOW
    openai = b'{"error": {"message": "This model\'s maximum context length is 8192 tokens. However, you requested 9000 tokens."}}'
    assert classify_upstream_error(400, openai) == OVERFLOW
    assert classify_upstream_error(413, b'{"error": {"message": "payload too large"}}') == OVERFLOW


def test_quota_errors_still_classified_quota():
    quota = b'{"error": {"message": "You exceeded your current quota, please check your plan and billing details."}}'
    assert classify_upstream_error(400, quota) == QUOTA
    assert classify_upstream_error(429, b"{}") == QUOTA


def test_plain_client_error_still_request():
    body = b'{"error": {"message": "max_tokens must be positive"}}'
    assert classify_upstream_error(400, body) == REQUEST


def test_media_rejection_looks_like_detection():
    assert looks_like_media_rejection("Model does not support video input")
    assert looks_like_media_rejection("unsupported content type: audio/wav")
    assert not looks_like_media_rejection("max_tokens must be positive")
    assert not looks_like_media_rejection("image is too large")  # no rejection phrase
