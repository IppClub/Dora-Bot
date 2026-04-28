from pathlib import Path

import pytest

from dora_ops.group_chat import GroupBufferedMessage, GroupMessageInput
from dora_ops.runtime import DoraOpsRuntime


CONFIG = """
paths:
  data_dir: data
  mirror_dir: mirrors
  job_dir: jobs
  summary_dir: summaries
admin:
  user_ids: [123]
  group_ids: []
group_chat:
  enabled: true
  enabled_group_ids: [456]
  bot_aliases: [多萝, Dora]
  acknowledge_feedback: true
  daily_group_analysis_limit: 1
  daily_user_analysis_limit: 1
  auto_create_analysis_jobs: false
repositories:
  dora_ssr:
    name: Dora SSR
    remote: https://gitcode.com/ippclub/Dora-SSR.git
    default_branch: main
    watch_tags: true
    watch_paths: []
  yuescript:
    name: YueScript
    remote: https://gitcode.com/ippclub/YueScript.git
    default_branch: main
    watch_tags: true
    watch_paths: []
"""


LLM_CONFIG = CONFIG + """
llm:
  enabled: true
  max_context_messages: 3
  chat:
    provider: openai-compatible
    base_url: https://example.invalid
    api_key_env: TEST_API_KEY
    model: test-model
    temperature: 0.1
    max_tokens: 128
    timeout_seconds: 5
"""


class FakeChatClient:
    def __init__(self, replies: list[str]):
        self.replies = replies
        self.calls: list[list[dict[str, str]]] = []

    async def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return self.replies.pop(0) if self.replies else ""


class FakeClassifierClient:
    def __init__(self, result: dict[str, object]):
        self.result = result
        self.calls: list[list[dict[str, str]]] = []

    async def complete_tool_call(self, messages, *, tool, tool_name):
        assert tool_name == "classify_message"
        self.calls.append(messages)
        return self.result


@pytest.mark.asyncio
async def test_group_feedback_is_recorded_and_acknowledged(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)

    result = await runtime.handle_group_message(
        GroupMessageInput(
            group_id=456,
            user_id=789,
            nickname="tester",
            text="Dora SSR Web IDE 创建文件后无法刷新",
        )
    )

    assert result is not None
    assert result.feedback_id is not None
    assert result.approval_id is not None
    assert result.mention_admin_id == 123
    assert result.reply is not None
    assert "已记录" in result.reply
    assert f"/approve feedback {result.feedback_id}" in result.reply
    stored = await runtime.storage.get_feedback(result.feedback_id)
    assert stored is not None
    assert stored["project"] == "Dora-SSR"
    approval = await runtime.storage.get_pending_approval("feedback", result.feedback_id)
    assert approval is not None
    assert approval["command"] == f"/approve feedback {result.feedback_id}"


@pytest.mark.asyncio
async def test_group_ignores_unrelated_and_disabled_groups(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)

    unrelated = await runtime.handle_group_message(
        GroupMessageInput(group_id=456, user_id=789, nickname="tester", text="今天吃什么")
    )
    assert unrelated is None

    disabled_group = await runtime.handle_group_message(
        GroupMessageInput(group_id=999, user_id=789, nickname="tester", text="Dora SSR 报错")
    )
    assert disabled_group is None


@pytest.mark.asyncio
async def test_group_alias_reply_without_recording_unrelated(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)

    result = await runtime.handle_group_message(
        GroupMessageInput(group_id=456, user_id=789, nickname="tester", text="多萝你怎么看这个")
    )
    assert result is not None
    assert result.feedback_id is None
    assert result.reply is not None
    assert "不归档" in result.reply


@pytest.mark.asyncio
async def test_group_at_only_wakes_bot(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)

    result = await runtime.handle_group_message(
        GroupMessageInput(group_id=456, user_id=789, nickname="tester", text="", mentions_bot=True)
    )

    assert result is not None
    assert result.reply is not None
    assert "不归档" in result.reply


@pytest.mark.asyncio
async def test_group_ignores_test_commands(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)

    result = await runtime.handle_group_message(
        GroupMessageInput(group_id=456, user_id=123, nickname="admin", text="/test ping")
    )
    assert result is None


@pytest.mark.asyncio
async def test_group_chat_llm_can_reply_when_mentioned(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(LLM_CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)
    fake = FakeChatClient(["只是刚好知道而已，别误会。"])
    runtime.group_chat.chat_client = fake

    result = await runtime.handle_group_message(
        GroupMessageInput(group_id=456, user_id=789, nickname="tester", text="多萝，今天聊点啥")
    )

    assert result is not None
    assert result.reason == "llm_chat"
    assert result.reply == "只是刚好知道而已，别误会。"
    assert fake.calls
    system_prompt = fake.calls[0][0]["content"]
    assert "# 角色设定" in system_prompt
    assert "坏酷又讨人喜欢的小萝莉" in system_prompt
    assert "# 多萝能做什么" in system_prompt
    assert "/test daily-summary --progress" not in system_prompt
    assert "/approve feedback <id>" not in system_prompt
    assert "不能假装已经执行命令" in system_prompt
    assert "# 能力触发规则" in system_prompt
    assert "报错、崩溃、无法、不能、失败" in system_prompt
    assert "追问报错全文、平台、版本、复现步骤" in system_prompt
    assert "消息中明确 @多萝 或提及 Dora SSR 就一定要回复" in system_prompt
    assert "不需要回复时返回空字符串" in system_prompt
    recent = await runtime.storage.list_recent_chat_messages("group:456", 10)
    assert [row["role"] for row in recent] == ["user", "assistant"]
    assert "tester：" in recent[0]["content"]


@pytest.mark.asyncio
async def test_group_chat_llm_empty_response_stays_silent(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(LLM_CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)
    fake = FakeChatClient([""])
    runtime.group_chat.chat_client = fake

    result = await runtime.handle_group_message(
        GroupMessageInput(group_id=456, user_id=789, nickname="tester", text="今天吃什么")
    )

    assert result is None
    assert fake.calls == []


@pytest.mark.asyncio
async def test_group_chat_uses_llm_classifier_for_project_question(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(LLM_CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)
    chat = FakeChatClient(["这个先看渲染状态切换和资源生命周期，别一上来就怀疑显卡。"])
    classifier = FakeClassifierClient(
        {
            "should_accept": False,
            "kind": "project_question",
            "project": "Dora-SSR",
            "confidence": 0.87,
            "needs_repo_analysis": False,
            "summary": "询问渲染管线拆分",
        }
    )
    runtime.group_chat.chat_client = chat
    runtime.group_chat.classifier_client = classifier

    result = await runtime.handle_group_message(
        GroupMessageInput(group_id=456, user_id=789, nickname="tester", text="这个帧管线怎么拆比较稳")
    )

    assert result is not None
    assert result.reason == "llm_chat"
    assert result.feedback_id is None
    assert result.classification.kind == "project_question"
    assert result.reply == "这个先看渲染状态切换和资源生命周期，别一上来就怀疑显卡。"
    assert classifier.calls
    assert chat.calls


@pytest.mark.asyncio
async def test_group_feedback_ack_takes_priority_over_llm_chat(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(LLM_CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)
    fake = FakeChatClient(["不应该使用这条回复"])
    runtime.group_chat.chat_client = fake

    result = await runtime.handle_group_message(
        GroupMessageInput(group_id=456, user_id=789, nickname="tester", text="Dora SSR Web IDE 创建文件后无法刷新")
    )

    assert result is not None
    assert result.reason == "manual_required"
    assert result.reply is not None
    assert "已记录" in result.reply
    assert fake.calls == []


@pytest.mark.asyncio
async def test_group_buffered_messages_are_stored_separately(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(LLM_CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)
    chat = FakeChatClient(["这个像资源生命周期没收好，先看释放点。"])
    classifier = FakeClassifierClient(
        {
            "should_accept": False,
            "kind": "project_question",
            "action": "answer_question",
            "project": "Dora-SSR",
            "confidence": 0.91,
            "needs_repo_analysis": False,
            "summary": "询问资源释放",
        }
    )
    runtime.group_chat.chat_client = chat
    runtime.group_chat.classifier_client = classifier

    result = await runtime.handle_group_message(
        GroupMessageInput(
            group_id=456,
            user_id=790,
            nickname="b",
            text="补充一下",
            mentions_bot=True,
            buffered_messages=(
                GroupBufferedMessage(789, "a", "多萝，Dora SSR 资源释放有问题", True),
                GroupBufferedMessage(790, "b", "场景切换后显存没下来", False),
            ),
        )
    )

    assert result is not None
    assert result.reply == "这个像资源生命周期没收好，先看释放点。"
    assert classifier.calls
    assert "a: 多萝，Dora SSR 资源释放有问题" in classifier.calls[0][1]["content"]
    recent = await runtime.storage.list_recent_chat_messages("group:456", 10)
    assert [row["role"] for row in recent] == ["user", "user", "assistant"]
    assert "a：" in recent[0]["content"]
    assert "b：" in recent[1]["content"]
