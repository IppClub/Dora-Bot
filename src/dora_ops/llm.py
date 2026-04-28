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

    async def complete_tool_call(
        self,
        messages: list[dict[str, str]],
        *,
        tool: dict[str, Any],
        tool_name: str,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self._complete_tool_call_sync, messages, tool, tool_name)

    def _complete_sync(self, messages: list[dict[str, str]]) -> str:
        message = self._request_message(
            {
                "model": self.profile.model,
                "messages": messages,
                "temperature": self.profile.temperature,
                "max_tokens": self.profile.max_tokens,
            }
        )
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMError("LLM 响应内容为空")
        return content.strip()

    def _complete_tool_call_sync(
        self,
        messages: list[dict[str, str]],
        tool: dict[str, Any],
        tool_name: str,
    ) -> dict[str, Any]:
        message = self._request_message(
            {
                "model": self.profile.model,
                "messages": messages,
                "temperature": self.profile.temperature,
                "max_tokens": self.profile.max_tokens,
                "tools": [tool],
                "tool_choice": {"type": "function", "function": {"name": tool_name}},
            }
        )
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            function = tool_calls[0].get("function")
            if isinstance(function, dict):
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    value = json.loads(arguments)
                    if isinstance(value, dict):
                        return value
        function_call = message.get("function_call")
        if isinstance(function_call, dict):
            arguments = function_call.get("arguments")
            if isinstance(arguments, str):
                value = json.loads(arguments)
                if isinstance(value, dict):
                    return value
        raise LLMError("LLM 响应缺少工具调用参数")

    def _request_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        api_key = os.environ.get(self.profile.api_key_env)
        if not api_key:
            raise LLMError(f"缺少环境变量：{self.profile.api_key_env}")
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
        return message
