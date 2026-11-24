from __future__ import annotations

import pytest

from dora_ops.config import LLMProfileConfig
from dora_ops.llm import LLMError, OpenAICompatibleChatClient


def test_complete_retries_empty_content_then_returns_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OpenAICompatibleChatClient(LLMProfileConfig())
    responses = iter([({"content": ""}, "length"), ({"content": "  中文摘要  "}, "stop")])
    monkeypatch.setattr(client, "_request_choice", lambda payload: next(responses))
    monkeypatch.setattr("dora_ops.llm.time.sleep", lambda seconds: None)

    assert client._complete_sync([{"role": "user", "content": "test"}]) == "中文摘要"


def test_complete_reports_empty_content_after_bounded_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OpenAICompatibleChatClient(LLMProfileConfig())
    calls = 0

    def empty_response(payload: object) -> tuple[dict[str, str], str]:
        nonlocal calls
        calls += 1
        return {"content": ""}, "length"

    monkeypatch.setattr(client, "_request_choice", empty_response)
    monkeypatch.setattr("dora_ops.llm.time.sleep", lambda seconds: None)

    with pytest.raises(LLMError, match=r"已重试 3 次.*finish_reason=length"):
        client._complete_sync([{"role": "user", "content": "test"}])
    assert calls == 3


def test_tool_call_uses_thinking_compatible_choice_and_requires_arguments(monkeypatch):
    client = OpenAICompatibleChatClient(LLMProfileConfig())
    payloads = []
    def response(payload):
        payloads.append(payload)
        return {"tool_calls": [{"function": {"name": "classify_message", "arguments": '{"action":"reply"}'}}]}
    monkeypatch.setattr(client, "_request_message", response)
    result = client._complete_tool_call_sync([], {"type": "function", "function": {"name": "classify_message"}}, "classify_message")
    assert payloads[0]["tool_choice"] == "auto"
    assert result == {"action": "reply"}
    monkeypatch.setattr(client, "_request_message", lambda payload: {"content": "普通正文不是分类结果"})
    with pytest.raises(LLMError, match="缺少工具调用参数"):
        client._complete_tool_call_sync([], {}, "classify_message")
