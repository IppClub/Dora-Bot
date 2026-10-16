import pytest

from dora_ops.config import LLMConfig
from dora_ops.llm import LLMError
from dora_ops.vision import VisionService


@pytest.mark.asyncio
async def test_vision_sends_multimodal_request_and_bounds_images(monkeypatch):
    service = VisionService(LLMConfig(enabled=True, vision={"max_images": 2}))
    calls = []

    async def complete(messages):
        calls.append(messages)
        return "截图显示模块加载失败。"

    monkeypatch.setattr(service.client, "complete", complete)
    result = await service.describe("哪里错了？", [
        "https://example.org/a.png", "https://example.org/a.png",
        "https://example.org/b.png", "https://example.org/c.png",
    ])
    content = calls[0][-1]["content"]
    assert content[0]["text"] == "随附文字：哪里错了？"
    assert [item["image_url"]["url"] for item in content[1:]] == [
        "https://example.org/a.png", "https://example.org/b.png",
    ]
    assert "截图显示模块加载失败。" in result
    assert "前 2 张" in result


@pytest.mark.asyncio
async def test_vision_failure_preserves_question_without_claiming_image_read(monkeypatch):
    service = VisionService(LLMConfig(enabled=True))

    async def complete(messages):
        raise LLMError("unavailable")

    monkeypatch.setattr(service.client, "complete", complete)
    result = await service.describe("帮我看报错", ["https://example.org/a.png"])
    assert "帮我看报错" in result
    assert "本次读取失败" in result
    assert "https://" not in result


@pytest.mark.asyncio
async def test_vision_disabled_and_invalid_sources_do_not_call_api():
    service = VisionService(LLMConfig(enabled=False))
    assert await service.describe("普通消息", []) == "普通消息"
    assert "未能读取" in await service.describe("", ["file:///etc/passwd"])
