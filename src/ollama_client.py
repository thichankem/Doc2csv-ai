"""Minimal Ollama HTTP client with optional token streaming."""
import json
from typing import Callable, Optional

import requests

DEFAULT_BASE = "http://localhost:11434"


def is_running(base_url: str = DEFAULT_BASE, timeout: float = 2.0) -> bool:
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=timeout)
        return r.status_code == 200
    except requests.RequestException:
        return False


def list_models(base_url: str = DEFAULT_BASE, timeout: float = 5.0) -> list[str]:
    """Return names of locally installed Ollama models, or [] if unreachable."""
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return [m["name"] for m in data.get("models", []) if "name" in m]
    except requests.RequestException:
        return []


def generate(
    model: str,
    prompt: str,
    base_url: str = DEFAULT_BASE,
    temperature: float = 0.3,
    num_ctx: int = 8192,
    timeout: int = 600,
    on_token: Optional[Callable[[str, int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> str:
    """Call /api/generate with streaming. Returns the full response text.

    - on_token(chunk_text, total_chars): invoked for each streamed chunk; use it
      to update UI without blocking.
    - should_stop(): if it returns True, the stream is aborted and a partial
      response (whatever was generated so far) is returned.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
        },
    }

    parts: list[str] = []
    total = 0

    with requests.post(
        f"{base_url}/api/generate",
        json=payload,
        timeout=timeout,
        stream=True,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if should_stop is not None and should_stop():
                break
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            piece = data.get("response", "")
            if piece:
                parts.append(piece)
                total += len(piece)
                if on_token is not None:
                    try:
                        on_token(piece, total)
                    except Exception:
                        pass
            if data.get("done"):
                break

    return "".join(parts)
