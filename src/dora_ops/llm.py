from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from typing import Any

from .config import LLMProfileConfig


class LLMError(RuntimeError):
    pass


class OpenAICompatibleChatClient:
    def __init__(self, profile: LLMProfileConfig):
        self.profile = profile

    async def complete(self, messages: list[dict[str, str]]) -> str:
        return await asyncio.to_thread(self._complete_sync, messages)

    def _complete_sync(self, messages: list[dict[str, str]]) -> str:
        api_key = os.environ.get(self.profile.api_key_env)
        if not api_key:
            raise LLMError(f"缺少环境变量：{self.profile.api_key_env}")
        payload = {
            "model": self.profile.model,
            "messages": messages,
            "temperature": self.profile.temperature,
            "max_tokens": self.profile.max_tokens,
        }
        url = f"{self.profile.base_url.rstrip('/')}/chat/completions"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.profile.timeout_seconds) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise LLMError(f"LLM 请求失败：HTTP {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"LLM 请求失败：{exc}") from exc

        data: dict[str, Any] = json.loads(raw)
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMError("LLM 响应缺少 choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise LLMError("LLM 响应缺少 message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMError("LLM 响应内容为空")
        return content.strip()
