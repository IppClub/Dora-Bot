from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from .config import LLMProfileConfig


class LLMError(RuntimeError):
    pass


class OpenAICompatibleChatClient:
    _EMPTY_CONTENT_ATTEMPTS = 3
    _EMPTY_CONTENT_RETRY_SECONDS = 0.5

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
        payload = {
            "model": self.profile.model,
            "messages": messages,
            "temperature": self.profile.temperature,
            "max_tokens": self.profile.max_tokens,
        }
        finish_reason = ""
        for attempt in range(self._EMPTY_CONTENT_ATTEMPTS):
            message, finish_reason = self._request_choice(payload)
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if attempt + 1 < self._EMPTY_CONTENT_ATTEMPTS:
                time.sleep(self._EMPTY_CONTENT_RETRY_SECONDS * (attempt + 1))
        raise LLMError(
            "LLM 响应内容为空"
            f"（已重试 {self._EMPTY_CONTENT_ATTEMPTS} 次，finish_reason={finish_reason or 'unknown'}）；"
            "若 finish_reason=length，说明推理模型的思考耗尽了 max_tokens，需调大该 profile 的 max_tokens"
        )

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
        message, _ = self._request_choice(payload)
        return message

    def _request_choice(self, payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
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
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise LLMError("LLM 响应缺少 message")
        return message, str(choice.get("finish_reason") or "")
