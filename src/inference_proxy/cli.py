"""Claude and Codex command-line apps as providers.

The app is run once per request with the conversation on stdin, and its reply is
wrapped as an OpenAI chat completion (one SSE chunk when the client streams).
Each CLI is a single login, so a CLI provider has one account. Tool calls and
token-by-token streaming are not available through a CLI.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import httpx

CLI_TYPES = ("claude", "codex")
KNOWN_MODELS = {
    "claude": ["sonnet", "opus", "haiku"],
    "codex": ["gpt-5", "gpt-5-codex", "o4-mini"],
}

_QUOTA = re.compile(r"usage limit|rate limit|quota|credit|too many requests|\b429\b|limit reached", re.I)
_AUTH = re.compile(r"not logged in|log in|login|authenticat|unauthori[sz]ed|api key|\b401\b", re.I)
_MODEL = re.compile(r"model.*(not found|does not exist|invalid|unknown|not available)|unknown model|invalid model", re.I)
_DATA_URL = re.compile(r"^data:(image/[a-zA-Z+.-]+);base64,(.*)$", re.S)


class _Bytes(httpx.AsyncByteStream):
    def __init__(self, data: bytes) -> None:
        self.data = data

    async def __aiter__(self):
        yield self.data

    async def aclose(self) -> None:
        return None


def _response(status: int, content_type: str, data: bytes) -> httpx.Response:
    return httpx.Response(status, headers={"content-type": content_type}, stream=_Bytes(data))


def _error(status: int, message: str) -> httpx.Response:
    return _response(status, "application/json", json.dumps({"error": {"message": message}}).encode())


def forget_session(tool: str, workdir: str) -> None:
    """Remove the session record a Claude call leaves behind. Codex runs with --ephemeral and writes none.

    Each call runs in its own temporary folder, and Claude files its session under a project named after that folder,
    so deleting that project removes the whole trace.
    """
    if tool != "claude":
        return
    project = Path.home() / ".claude" / "projects" / re.sub(r"[^A-Za-z0-9]", "-", os.path.realpath(workdir))
    shutil.rmtree(project, ignore_errors=True)


def conversation(messages: list[dict], image_dir: Path) -> tuple[str, list[Path]]:
    """Flatten chat messages to one prompt. Image data URLs are written to files."""
    images: list[Path] = []
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).capitalize()
        content = message.get("content")
        parts = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
        text = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                text.append(str(part.get("text", "")))
            elif part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                match = _DATA_URL.match(url)
                if not match:
                    raise ValueError("image not supported: only inline (data URL) images work through a CLI provider")
                path = image_dir / f"image{len(images) + 1}.{match.group(1).split('/')[1].split('+')[0]}"
                path.write_bytes(base64.b64decode(match.group(2)))
                images.append(path)
                text.append(f"[image {len(images)}: {path.name}]")
            else:
                raise ValueError(f"content part '{part.get('type')}' not supported through a CLI provider")
        lines.append(f"{role}: " + "\n".join(text))
    prompt = ("Continue this conversation. Reply with only the assistant's next message, "
              "with no preamble about the conversation itself.\n\n" + "\n\n".join(lines))
    return prompt, images


def command(tool: str, model: str | None, images: list[Path]) -> list[str]:
    if tool == "codex":
        cmd = ["codex", "exec", "--skip-git-repo-check", "--ephemeral", "--sandbox", "read-only"]
        if model:
            cmd += ["-m", model]
        for image in images:
            cmd += ["-i", str(image)]
        return cmd
    cmd = ["claude", "-p", "--output-format", "text"]
    if model:
        cmd += ["--model", model]
    if images:
        cmd += ["--allowedTools", "Read"]
    return cmd


def _completion(model: str, text: str, stream: bool) -> httpx.Response:
    ident, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())
    if not stream:
        body = {"id": ident, "object": "chat.completion", "created": created, "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}
        return _response(200, "application/json", json.dumps(body).encode())
    def chunk(delta: dict, finish: str | None = None) -> str:
        return "data: " + json.dumps({"id": ident, "object": "chat.completion.chunk", "created": created, "model": model,
                                      "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}) + "\n\n"
    data = chunk({"role": "assistant", "content": ""}) + chunk({"content": text}) + chunk({}, "stop") + "data: [DONE]\n\n"
    return _response(200, "text/event-stream", data.encode())


async def ask_cli(tool: str, prompt: str, model: str | None = None, timeout: float = 300.0, web: bool = False) -> str:
    """One prompt in, text out. With web=True the Claude CLI may use WebSearch and WebFetch (nothing else)."""
    if not shutil.which(tool):
        raise RuntimeError(f"{tool} is not installed or not on PATH")
    if web and tool != "claude":
        raise ValueError("web research needs the claude provider")
    cmd = command(tool, model, [])
    if web:
        cmd += ["--allowedTools", "WebSearch", "WebFetch"]
    with tempfile.TemporaryDirectory(prefix="inference-proxy-") as tmp:
        proc = await asyncio.create_subprocess_exec(*cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.PIPE, cwd=tmp)
        try:
            out, err = await asyncio.wait_for(proc.communicate(prompt.encode()), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(f"{tool} did not answer within {timeout:.0f}s") from None
        finally:
            forget_session(tool, tmp)
    if proc.returncode:
        raise RuntimeError((err.decode("utf-8", "replace") or out.decode("utf-8", "replace")).strip()[:300] or f"{tool} exited with {proc.returncode}")
    return out.decode("utf-8", "replace").strip()


async def run_cli(tool: str, model: str | None, body: dict, timeout: float, stream: bool) -> httpx.Response:
    if not shutil.which(tool):
        return _error(503, f"{tool} is not installed or not on PATH")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return _error(400, "'messages' must be a non-empty list")
    with tempfile.TemporaryDirectory(prefix="inference-proxy-") as tmp:
        try:
            prompt, images = conversation(messages, Path(tmp))
        except ValueError as exc:
            return _error(400, str(exc))
        try:
            proc = await asyncio.create_subprocess_exec(
                *command(tool, model, images), stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=tmp)
            out, err = await asyncio.wait_for(proc.communicate(prompt.encode()), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return _error(504, f"{tool} did not answer within {timeout:.0f}s")
        except OSError as exc:
            return _error(503, f"could not start {tool}: {exc}")
        finally:
            forget_session(tool, tmp)
    text, errtext = out.decode("utf-8", "replace").strip(), err.decode("utf-8", "replace").strip()
    if proc.returncode or not text:
        detail = (errtext or text or f"{tool} exited with {proc.returncode}")[:300]
        for pattern, status in ((_QUOTA, 429), (_AUTH, 401), (_MODEL, 404)):
            if pattern.search(detail):
                return _error(status, detail)
        return _error(500, detail)
    return _completion(model or tool, text, stream)
