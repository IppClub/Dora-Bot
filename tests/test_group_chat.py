from pathlib import Path

import pytest

from dora_ops.group_chat import GroupBufferedMessage, GroupMention, GroupMessageInput, GroupMessageService
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
  auto_analysis_24h_limit: 0
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


def test_group_project_route_prefers_dora_ssr_for_combined_project() -> None:
    assert GroupMessageService._repo_key_for_project("Dora-SSR/YueScript") == "dora_ssr"
    assert GroupMessageService._repo_key_for_project("YueScript") == "yuescript"


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
    assert result.mention_admin_id is None
    assert result.reply is None
    assert result.admin_notification is not None
    assert "群聊反馈已记录：#" in result.admin_notification
    assert f"/approve feedback {result.feedback_id}" in result.admin_notification
    stored = await runtime.storage.get_feedback(result.feedback_id)
    assert stored is not None
    assert stored["project"] == "Dora-SSR"
    approval = await runtime.storage.get_pending_approval("feedback", result.feedback_id)
    assert approval is not None
    assert approval["command"] == f"/approve feedback {result.feedback_id}"


@pytest.mark.asyncio
async def test_group_feedback_auto_creates_analysis_job_within_daily_limit(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(CONFIG.replace("auto_analysis_24h_limit: 0", "auto_analysis_24h_limit: 10"), encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)

    async def fake_ensure_mirror(repo_key, repo):
        return tmp_path

    async def fake_create_feedback_analysis(repo_key, repo_path, feedback_id, prompt_text, *, triggered_by=None, trigger_source="admin_approval"):
        assert repo_key == "dora_ssr"
        assert repo_path == tmp_path
        assert feedback_id == 1
        assert triggered_by == "789"
        assert trigger_source == "group_auto_analysis"
        return await runtime.storage.create_job(
            kind="feedback_analysis",
            target_type="feedback",
            target_id=feedback_id,
            tmux_session="queued_feedback",
            prompt_path=tmp_path / "prompt.md",
            output_path=tmp_path / "output.json",
            error_path=tmp_path / "error.log",
            exit_code_path=tmp_path / "exit_code",
            done_path=tmp_path / "done",
            triggered_by=triggered_by,
            trigger_source="group_auto_analysis",
        )

    runtime.group_chat.tracker.ensure_mirror = fake_ensure_mirror  # type: ignore[method-assign]
    runtime.group_chat.jobs.create_feedback_analysis = fake_create_feedback_analysis  # type: ignore[method-assign]

    result = await runtime.handle_group_message(
        GroupMessageInput(group_id=456, user_id=789, nickname="tester", text="Dora SSR 渲染管线怎么拆")
    )

    assert result is not None
    assert result.reason == "auto_accepted"
    assert result.analysis_job_id is not None
    assert result.approval_id is None
    assert "已自动放行" in (result.admin_notification or "")
    assert await runtime.storage.get_pending_approval("feedback", result.feedback_id or 0) is None


@pytest.mark.asyncio
async def test_group_feedback_falls_back_to_approval_after_daily_auto_limit(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(CONFIG.replace("auto_analysis_24h_limit: 0", "auto_analysis_24h_limit: 1"), encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)
    job_file = tmp_path / "job_file"
    job_file.write_text("", encoding="utf-8")
    await runtime.storage.create_job(
        kind="feedback_analysis",
        target_type="feedback",
        target_id=99,
        tmux_session="already_used",
        prompt_path=job_file,
        output_path=job_file,
        error_path=job_file,
        exit_code_path=job_file,
        done_path=job_file,
    )

    result = await runtime.handle_group_message(
        GroupMessageInput(group_id=456, user_id=789, nickname="tester", text="Dora SSR 渲染管线怎么拆")
    )

    assert result is not None
    assert result.reason == "manual_required"
    assert result.analysis_job_id is None
    assert result.approval_id is not None
    assert result.admin_notification is not None
    assert "/approve feedback" in result.admin_notification


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
    assert "不要直接解释；应记录为项目问题" in system_prompt
    assert "消息中明确 @多萝 且不是 Dora SSR/YueScript/游戏引擎项目问题" in system_prompt
    assert "不要把夸奖、感谢或调侃理解成对自己的评价" in system_prompt
    assert "不需要回复时返回空字符串" in system_prompt
    recent = await runtime.storage.list_recent_chat_messages("group:456", 10)
    assert [row["role"] for row in recent] == ["user", "assistant"]
    assert "tester(QQ:789)：" in recent[0]["content"]
    assert fake.calls[0][-1]["role"] == "user"
    assert "# 最新群聊消息" in fake.calls[0][-1]["content"]
    assert "tester(QQ:789)：多萝，今天聊点啥" in fake.calls[0][-1]["content"]


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
async def test_group_chat_llm_final_user_message_is_latest_message(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(LLM_CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)
    await runtime.storage.append_chat_message("group:456", "user", "H_PIGCATDOG(QQ:111)：简单介绍一下Dora引擎里TSTL")
    await runtime.storage.append_chat_message("group:456", "assistant", "TSTL 就是 Dora SSR 里把 TS/TSX 转成 Lua 的组件。")
    fake = FakeChatClient(["你这句 at 了 H_PIGCATDOG，不是在问 TSTL。"])
    runtime.group_chat.chat_client = fake

    result = await runtime.handle_group_message(
        GroupMessageInput(
            group_id=456,
            user_id=111,
            nickname="H_PIGCATDOG",
            text="@多萝(QQ: 3844055063) 你能看到我在 at 谁吗？告诉我你看到我这句话at的人。 @H_PIGCATDOG(QQ: 111)",
            mentions_bot=True,
            mentions=(GroupMention("3844055063", "多萝"), GroupMention("111", "H_PIGCATDOG")),
        )
    )

    assert result is not None
    assert result.reply == "你这句 at 了 H_PIGCATDOG，不是在问 TSTL。"
    assert fake.calls
    llm_messages = fake.calls[0]
    assert llm_messages[-1]["role"] == "user"
    assert "# 最新群聊消息" in llm_messages[-1]["content"]
    assert "你能看到我在 at 谁吗" in llm_messages[-1]["content"]
    assert "@H_PIGCATDOG(QQ: 111)" in llm_messages[-1]["content"]
    assert "TSTL" not in llm_messages[-1]["content"]


@pytest.mark.asyncio
async def test_group_chat_records_project_question_for_repo_analysis(tmp_path: Path) -> None:
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
    assert result.reason == "manual_required"
    assert result.feedback_id is not None
    assert result.approval_id is not None
    assert result.classification.kind == "project_question"
    assert result.classification.needs_repo_analysis is True
    assert result.reply is None
    assert result.admin_notification is not None
    assert "群聊反馈已记录：#" in result.admin_notification
    assert classifier.calls
    assert chat.calls == []


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
    assert result.reply is None
    assert result.admin_notification is not None
    assert "群聊反馈已记录：#" in result.admin_notification
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
    assert result.reply is None
    assert result.feedback_id is not None
    assert result.approval_id is not None
    assert result.admin_notification is not None
    assert classifier.calls
    assert "a(QQ:789): 多萝，Dora SSR 资源释放有问题" in classifier.calls[0][1]["content"]
    recent = await runtime.storage.list_recent_chat_messages("group:456", 10)
    assert [row["role"] for row in recent] == ["user", "user"]
    assert "a(QQ:789)：" in recent[0]["content"]
    assert "b(QQ:790)：" in recent[1]["content"]


@pytest.mark.asyncio
async def test_group_chat_history_preserves_non_bot_mentions_as_targets(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(LLM_CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)
    chat = FakeChatClient([""])
    runtime.group_chat.chat_client = chat

    result = await runtime.handle_group_message(
        GroupMessageInput(
            group_id=456,
            user_id=789,
            nickname="我心飞翔",
            text="@yuueang 大佬的行动力太惊人了",
            mentions_bot=False,
            mentions=(GroupMention("111", "yuueang"),),
        )
    )

    assert result is None
    recent = await runtime.storage.list_recent_chat_messages("group:456", 10)
    assert recent[0]["content"] == "我心飞翔(QQ:789)：@yuueang 大佬的行动力太惊人了 [mentions: @yuueang(QQ: 111)]"
