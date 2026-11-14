import json
import os

import pytest

from inference_proxy.cli import run_cli


def install_cli(tmp_path, monkeypatch, tool, *, stdout="hello there", stderr="", code=0, sleep=0):
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    argv_file, stdin_file = tmp_path / "argv", tmp_path / "stdin"
    script = bindir / tool
    script.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$@\" > {argv_file}\n"
        f"cat > {stdin_file}\n"
        f"sleep {sleep}\n"
        f"printf '%s' {json.dumps(stdout)}\n"
        f"printf '%s' {json.dumps(stderr)} >&2\n"
        f"exit {code}\n"
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ.get('PATH', '')}")
    return argv_file, stdin_file


def body(content="Please answer"):
    return {"messages": [{"role": "user", "content": content}]}


async def read_json(response):
    await response.aread()
    return json.loads(response.content)


@pytest.mark.parametrize("tool", ["claude", "codex"])
async def test_normal_reply(tool, tmp_path, monkeypatch):
    _, stdin_file = install_cli(tmp_path, monkeypatch, tool)
    response = await run_cli(tool, None, body(), 2, False)
    data = await read_json(response)
    assert response.status_code == 200
    assert data["choices"][0]["message"]["content"] == "hello there"
    assert data["object"] == "chat.completion"
    assert "Please answer" in stdin_file.read_text()


async def test_stream_response(tmp_path, monkeypatch):
    install_cli(tmp_path, monkeypatch, "claude")
    response = await run_cli("claude", None, body(), 2, True)
    await response.aread()
    text = response.content.decode()
    assert response.headers["content-type"] == "text/event-stream"
    assert any(json.loads(chunk[6:])["choices"][0]["delta"].get("content") == "hello there"
               for chunk in text.split("\n\n") if chunk.startswith("data: {") )
    assert any(json.loads(chunk[6:])["choices"][0]["finish_reason"] == "stop"
               for chunk in text.split("\n\n") if chunk.startswith("data: {"))
    assert text.endswith("data: [DONE]\n\n")


@pytest.mark.parametrize(("stderr", "status"), [
    ("usage limit reached", 429), ("Not logged in, please login", 401),
    ("model foo not found", 404), ("something went wrong", 500),
])
async def test_failure_mapping(stderr, status, tmp_path, monkeypatch):
    install_cli(tmp_path, monkeypatch, "claude", stdout="", stderr=stderr, code=1)
    response = await run_cli("claude", None, body(), 2, False)
    assert response.status_code == status


async def test_cli_missing_returns_503(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    response = await run_cli("claude", None, body(), 2, False)
    assert response.status_code == 503


async def test_timeout_returns_504(tmp_path, monkeypatch):
    install_cli(tmp_path, monkeypatch, "claude", sleep=1)
    response = await run_cli("claude", None, body(), 0.5, False)
    assert response.status_code == 504


@pytest.mark.parametrize("tool", ["claude", "codex"])
async def test_inline_image_is_written_and_cli_options_added(tool, tmp_path, monkeypatch):
    argv_file, _ = install_cli(tmp_path, monkeypatch, tool)
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}
    response = await run_cli(tool, None, body([{"type": "text", "text": "look"}, image]), 2, False)
    assert response.status_code == 200
    args = argv_file.read_text().splitlines()
    if tool == "codex":
        index = args.index("-i")
        assert (tmp_path / "nonexistent").exists() is False  # image is held in cli's temporary directory
        assert args[index + 1].endswith("image1.png")
    else:
        assert args[args.index("--allowedTools") + 1] == "Read"


async def test_remote_image_url_is_rejected(tmp_path, monkeypatch):
    install_cli(tmp_path, monkeypatch, "claude")
    image = {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}}
    response = await run_cli("claude", None, body([image]), 2, False)
    data = await read_json(response)
    assert response.status_code == 400
    assert "image" in data["error"]["message"].lower()
    assert "not support" in data["error"]["message"].lower()


@pytest.mark.parametrize(("tool", "flag"), [("claude", "--model"), ("codex", "-m")])
async def test_model_argument(tool, flag, tmp_path, monkeypatch):
    argv_file, _ = install_cli(tmp_path, monkeypatch, tool)
    response = await run_cli(tool, "haiku", body(), 2, False)
    args = argv_file.read_text().splitlines()
    assert response.status_code == 200
    assert args[args.index(flag) + 1] == "haiku"


async def test_empty_stdout_is_error(tmp_path, monkeypatch):
    install_cli(tmp_path, monkeypatch, "claude", stdout="")
    response = await run_cli("claude", None, body(), 2, False)
    assert response.status_code == 500


async def test_unsupported_content_part_is_rejected(tmp_path, monkeypatch):
    install_cli(tmp_path, monkeypatch, "claude")
    response = await run_cli("claude", None, body([{"type": "input_audio"}]), 2, False)
    assert response.status_code == 400
