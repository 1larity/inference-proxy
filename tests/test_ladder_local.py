from inference_proxy.ladder import is_local_url, suggest


def test_local_urls():
    for url in ("http://127.0.0.1:1234/v1", "http://localhost:8080", "http://192.168.1.125:1234/v1", "http://10.0.0.5/v1", "http://box.local/v1"):
        assert is_local_url(url)
    for url in ("https://api.z.ai/api/paas/v4", "https://openrouter.ai/api/v1", "http://8.8.8.8/v1", "not a url"):
        assert not is_local_url(url)


def test_local_model_beats_hosted_with_unknown_price_on_a_tie():
    local = {"provider": "l", "model": "small", "score": 140.0, "local": True}
    hosted = {"provider": "h", "model": "other", "score": 141.0}  # no price known
    best = suggest([local, hosted], rungs=1, tolerance=0.03)["rungs"][0]
    assert best["model"] == "small"


def test_known_price_still_decides_between_hosted_models():
    cheap = {"provider": "h", "model": "flash", "score": 153.8, "input_per_m": 1, "output_per_m": 1}
    dear = {"provider": "h", "model": "full", "score": 157.1, "input_per_m": 3, "output_per_m": 3}
    assert suggest([cheap, dear], rungs=1, tolerance=0.03)["rungs"][0]["model"] == "flash"
    assert suggest([cheap, dear], rungs=1, tolerance=0.0)["rungs"][0]["model"] == "full"
