"""
Thin client for a vLLM OpenAI-compatible server (/v1/chat/completions).
Tracks Time-To-First-Token (TTFT) and total latency when streaming.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Generator, List, Optional

import requests


@dataclass
class VLLMResponse:
    text: str
    ttft_ms: Optional[float]
    total_latency_ms: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class VLLMClient:
    """
    Works against either:
      - a self-hosted vLLM OpenAI-compatible server (no auth), or
      - NVIDIA NIM's hosted API (https://build.nvidia.com) which uses the
        same OpenAI-compatible schema but requires a Bearer API key.
    """

    def __init__(self, api_url: str, model_name: str, api_key: str | None = None, timeout_s: float = 120.0):
        self.api_url = api_url
        self.model_name = model_name
        self.api_key = api_key
        self.timeout_s = timeout_s

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 512,
        stream: bool = False,
    ) -> VLLMResponse:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        t0 = time.perf_counter()

        if not stream:
            resp = requests.post(self.api_url, json=payload, headers=headers, timeout=self.timeout_s)
            total_latency_ms = (time.perf_counter() - t0) * 1000
            if resp.status_code != 200:
                return VLLMResponse(
                    text=f"[vLLM error {resp.status_code}] {resp.text}",
                    ttft_ms=None,
                    total_latency_ms=total_latency_ms,
                )
            data = resp.json()
            usage = data.get("usage", {})
            return VLLMResponse(
                text=data["choices"][0]["message"]["content"],
                ttft_ms=None,
                total_latency_ms=total_latency_ms,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
            )

        # Streaming path: measure TTFT precisely.
        ttft_ms = None
        chunks: List[str] = []
        try:
            with requests.post(
                self.api_url, json=payload, headers=headers, timeout=self.timeout_s, stream=True
            ) as resp:
                for line in resp.iter_lines():
                    if not line:
                        continue
                    line = line.decode("utf-8").removeprefix("data: ").strip()
                    if line == "[DONE]":
                        break
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {}).get("content")
                    if delta:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t0) * 1000
                        chunks.append(delta)
        except Exception as e:
            return VLLMResponse(
                text=f"[vLLM streaming error] {e}",
                ttft_ms=None,
                total_latency_ms=(time.perf_counter() - t0) * 1000,
            )

        total_latency_ms = (time.perf_counter() - t0) * 1000
        return VLLMResponse(text="".join(chunks), ttft_ms=ttft_ms, total_latency_ms=total_latency_ms)

    def chat_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> Generator[str, None, None]:
        """
        Yields text deltas as they arrive, for use with st.write_stream() so
        the UI shows tokens live instead of waiting for the full response.
        """
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            with requests.post(
                self.api_url, json=payload, headers=headers, timeout=self.timeout_s, stream=True
            ) as resp:
                if resp.status_code != 200:
                    yield f"[vLLM error {resp.status_code}] {resp.text}"
                    return
                for line in resp.iter_lines():
                    if not line:
                        continue
                    line = line.decode("utf-8").removeprefix("data: ").strip()
                    if line == "[DONE]":
                        break
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {}).get("content")
                    if delta:
                        yield delta
        except Exception as e:
            yield f"[vLLM streaming error] {e}"
