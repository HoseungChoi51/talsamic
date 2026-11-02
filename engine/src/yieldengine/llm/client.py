"""Minimal client for any OpenAI-compatible chat endpoint: ollama (http://localhost:11434/v1),
llama.cpp server, vLLM. Set YIELDENGINE_LLM_URL and YIELDENGINE_LLM_MODEL, or pass them in."""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass


@dataclass
class ChatClient:
    url: str = os.environ.get("YIELDENGINE_LLM_URL", "http://localhost:11434/v1")
    model: str = os.environ.get("YIELDENGINE_LLM_MODEL", "qwen3:32b")
    api_key: str = os.environ.get("YIELDENGINE_LLM_KEY", "local")
    timeout: float = 600

    def chat(self, system: str, user: str, temperature: float = 0.0, seed: int | None = None,
             json_mode: bool = True) -> str:
        body = {"model": self.model, "temperature": temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if seed is not None:
            body["seed"] = seed
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(self.url.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]


def parse_json(text: str) -> dict:
    """Parse a JSON object from a model reply (tolerates <think> blocks and code fences)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("no JSON object in the reply")
    return json.loads(m.group(0))
