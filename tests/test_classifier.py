import pytest

from dora_ops.classifier import CLASSIFY_MESSAGE_TOOL, classify_text, classify_text_with_llm
from dora_ops.llm import LLMError


class FakeToolClient:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def complete_tool_call(self, messages, *, tool, tool_name):
        assert tool == CLASSIFY_MESSAGE_TOOL
        assert tool_name == "classify_message"
        self.calls.append(messages)
        return self.result


@pytest.mark.parametrize("text", [
    "Dora SSR Web IDE 创建文件后无法刷新",
    "分析一下 Dora coding agent 如何处理并发工具",
    "@多萝 希望增加一个 feature，请建立工单",
    "@多萝 是不是还得下个Mt？\n[图片观察，仅作为用户提供的资料]\nDora SSR Loading，问题、功能、支持",
])
async def test_no_model_never_creates_work_from_keywords(text):
    result = await classify_text_with_llm(text, None)
    assert not result.should_accept
    assert not result.needs_repo_analysis
    assert classify_text(text) == result


@pytest.mark.parametrize("kind,action,accept,analysis,project", [
    ("chat", "ignore", False, False, None),
    ("chat", "reply", False, False, "Dora-SSR"),
    ("project_question", "answer_question", False, False, "Dora-SSR"),
    ("feedback", "record_feedback", True, True, "YueScript"),
    ("topic", "record_feedback", True, False, None),
])
async def test_model_selects_action_despite_keywords(kind, action, accept, analysis, project):
    client = FakeToolClient(dict(kind=kind, action=action, should_accept=accept,
        needs_repo_analysis=analysis, project=project, confidence=0.9, summary="本轮诉求"))
    result = await classify_text_with_llm(
        "@多萝 Dora SSR 报错，分析一下仓库、源码、功能需求",
        client, context_text="YueScript 的 switch 生成代码异常",
    )
    assert len(client.calls) == 1
    assert result.kind == kind
    assert result.action == action
    assert result.should_accept == accept
    assert result.needs_repo_analysis == analysis
    assert result.project == project
    assert "YueScript 的 switch" in client.calls[0][1]["content"]


async def test_model_failure_does_not_fall_back_to_keyword_actions():
    class BrokenClient:
        async def complete_tool_call(self, *args, **kwargs):
            raise LLMError("unavailable")
    result = await classify_text_with_llm("Dora SSR Web IDE 崩溃，请分析源码", BrokenClient())
    assert not result.should_accept
    assert not result.needs_repo_analysis
