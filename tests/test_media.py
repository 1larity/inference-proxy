from inference_proxy.media import (
    CAPABILITIES,
    media_capabilities,
    reasoning_requested,
    required_capabilities,
)


def test_text_only_body():
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert required_capabilities(body) == frozenset({"text"})


def test_media_part_types_detected():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look at this"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            },
            {"role": "user", "content": [{"type": "video_url", "video_url": "https://v.test/x.mp4"}]},
        ]
    }
    assert required_capabilities(body) == frozenset({"text", "image", "video"})


def test_audio_and_file_parts_detected():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
                    {"type": "file", "file": {"file_id": "f1"}},
                ],
            }
        ]
    }
    assert required_capabilities(body) == frozenset({"text", "audio", "file"})


def test_string_and_missing_content_are_text_only():
    assert required_capabilities({"messages": [{"role": "user", "content": "hi"}]}) == frozenset({"text"})
    assert required_capabilities({"messages": [{"role": "assistant"}]}) == frozenset({"text"})
    assert required_capabilities({}) == frozenset({"text"})


def test_non_media_part_types_ignored():
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "future_thing"}]}]}
    assert required_capabilities(body) == frozenset({"text"})


def test_reasoning_flags():
    assert reasoning_requested({"reasoning_effort": "high"})
    assert reasoning_requested({"reasoning": {"effort": "low"}})  # dict without explicit off type
    assert reasoning_requested({"reasoning": True})
    assert reasoning_requested({"enable_thinking": True})
    assert reasoning_requested({"thinking": {"type": "enabled", "effort": "high"}})
    assert not reasoning_requested({})
    assert not reasoning_requested({"reasoning_effort": "none"})
    assert not reasoning_requested({"reasoning": {"type": "disabled"}})
    assert not reasoning_requested({"enable_thinking": False})
    assert not reasoning_requested({"thinking": {"type": "disabled"}})


def test_reasoning_required_when_requested():
    body = {
        "messages": [{"role": "user", "content": "prove something"}],
        "reasoning_effort": "high",
    }
    assert required_capabilities(body) == frozenset({"text", "reasoning"})


def test_media_and_reasoning_combine():
    body = {
        "messages": [{"role": "user", "content": [{"type": "video_url", "video_url": "x"}]}],
        "thinking": {"type": "enabled"},
    }
    assert required_capabilities(body) == frozenset({"text", "video", "reasoning"})


def test_media_capabilities_reports_non_text_sorted():
    assert media_capabilities(frozenset({"text", "image", "reasoning"})) == ["image", "reasoning"]
    assert media_capabilities(frozenset({"text"})) == []


def test_vocabulary():
    assert CAPABILITIES == ("text", "image", "video", "audio", "file", "reasoning")
