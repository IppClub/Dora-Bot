import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from dora_ops.admin import AdminCommands
from dora_ops.models import JobStatus
from dora_ops.runtime import DoraOpsRuntime


PLUGIN_PATH = Path(__file__).resolve().parents[1] / "plugins" / "dora_ops" / "main.py"
PLUGIN_SPEC = importlib.util.spec_from_file_location("dora_ops_plugin_main", PLUGIN_PATH)
assert PLUGIN_SPEC is not None
plugin_module = importlib.util.module_from_spec(PLUGIN_SPEC)
assert PLUGIN_SPEC.loader is not None
PLUGIN_SPEC.loader.exec_module(plugin_module)
DoraOpsPlugin = plugin_module.DoraOpsPlugin


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
  auto_analysis_24h_limit: 0
welcome:
  enabled: true
  enabled_group_ids: [456]
  message: |
    欢迎 {name} 加入 Dora 社区！
    QQ：{user_id}
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
  max_context_messages: 2
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
    def __init__(self, reply: str = "LLM 多轮回复"):
        self.reply = reply
        self.calls: list[list[dict[str, str]]] = []

    async def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return self.reply


class FakeClassifierClient:
    def __init__(self, result: dict[str, object]):
        self.result = result
        self.calls: list[list[dict[str, str]]] = []

    async def complete_tool_call(self, messages, *, tool, tool_name):
        assert tool_name == "classify_message"
        self.calls.append(messages)
        return self.result


class FakePlannerClient:
    def __init__(self, result: dict[str, object]):
        self.result = result
        self.calls: list[list[dict[str, str]]] = []

    async def complete_tool_call(self, messages, *, tool, tool_name):
        assert tool_name == "plan_feedback_analysis"
        self.calls.append(messages)
        return self.result


def test_admin_project_route_prefers_dora_ssr_for_combined_project() -> None:
    assert AdminCommands._repo_key_for_project("Dora-SSR/YueScript") == "dora_ssr"
    assert AdminCommands._repo_key_for_project("YueScript") == "yuescript"


def test_feedback_session_repo_key_parser_allows_underscored_repo_key() -> None:
    assert DoraOpsPlugin._repo_key_from_feedback_session("dora_job_1777477435_dora_ssr_feedback_46") == "dora_ssr"
    assert DoraOpsPlugin._repo_key_from_feedback_session("dora_job_1777477435_yuescript_feedback_47") == "yuescript"
    assert DoraOpsPlugin._repo_key_from_feedback_session("invalid") is None


class FakePlainText:
    def __init__(self, text: str):
        self.text = text


class FakeAt:
    type = "at"

    def __init__(self, user_id: int = 999, name: str = ""):
        self.user_id = str(user_id)
        self.name = name


class FakeMessageArray:
    text = "array text"

    def __iter__(self):
        return iter([FakeAt(999)])

    def filter_at(self):
        return [FakeAt(999)]

    def is_at(self, user_id):
        return str(user_id) == "999"


class FakeToDictText:
    def to_dict(self):
        return {"type": "text", "data": {"text": "dict text"}}


class FakeToDictAt:
    def to_dict(self):
        return {"type": "at", "data": {"qq": "111", "name": "yuueang"}}


class FakeQQAPI:
    def __init__(self):
        self.private_messages = []
        self.group_messages = []
        self.query = FakeQQQuery()

    async def post_private_msg(self, user_id, **kwargs):
        self.private_messages.append((user_id, kwargs))

    async def post_group_msg(self, group_id, **kwargs):
        self.group_messages.append((group_id, kwargs))


class FakeQQQuery:
    def __init__(self):
        self.members = {}
        self.calls = []

    async def get_group_member_info(self, group_id, user_id):
        self.calls.append((group_id, user_id))
        return self.members.get(
            str(user_id),
            SimpleNamespace(card="", nickname=""),
        )


class FakeDebounceRuntime:
    def __init__(self):
        self.config = SimpleNamespace(
            admin=SimpleNamespace(user_ids={123, 456}),
            group_chat=SimpleNamespace(debounce_seconds=0),
        )
        self.messages = []

    async def handle_group_message(self, msg):
        self.messages.append(msg)
        return SimpleNamespace(reply=None, admin_notification=None, mention_admin_id=None, reason="test")

    def render_welcome_message(self, member):
        return None


@pytest.mark.asyncio
async def test_admin_ping_and_classify(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)

    denied = await runtime.handle_admin_text("/test ping", user_id=999)
    assert denied == "没有管理员权限。"

    pong = await runtime.handle_admin_text("/test ping", user_id=123)
    assert pong is not None
    assert "pong" in pong

    classified = await runtime.handle_admin_text("/test classify YueScript switch 报错", user_id=123)
    assert classified is not None
    assert "YueScript" in classified

    group_pong = await runtime.handle_admin_text("/test ping", user_id=123, group_id=456)
    assert group_pong is None


@pytest.mark.asyncio
async def test_admin_progress_report_requires_local_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)

    result = await runtime.handle_admin_text("/test daily-summary --progress", user_id=123)
    assert result is not None
    assert "缺少" in result


def test_plugin_message_segments_support_ncatbot_objects() -> None:
    msg = SimpleNamespace(self_id="999", message=[FakePlainText("准备 "), FakePlainText("尝试一种模式"), FakeAt(999)])

    assert DoraOpsPlugin._message_text(msg) == "准备 尝试一种模式@QQ: 999"
    assert DoraOpsPlugin._extract_text(msg) == "准备 尝试一种模式"
    assert DoraOpsPlugin._mentions_bot(msg) is True
    assert DoraOpsPlugin._mentions(msg)[0].user_id == "999"


def test_plugin_message_text_prefers_raw_message() -> None:
    msg = SimpleNamespace(raw_message="/test help", text="", message=[FakePlainText("ignored")])

    assert DoraOpsPlugin._message_text(msg) == "/test help"


def test_plugin_message_segments_support_message_array_helpers() -> None:
    msg = SimpleNamespace(self_id="999", message=FakeMessageArray())

    assert DoraOpsPlugin._message_text(msg) == "@QQ: 999"
    assert DoraOpsPlugin._extract_text(msg) == "array text"
    assert DoraOpsPlugin._mentions_bot(msg) is True


def test_plugin_message_segments_support_to_dict_objects() -> None:
    msg = SimpleNamespace(message=[FakeToDictText(), FakeToDictAt()])

    assert DoraOpsPlugin._message_text(msg) == "dict text@yuueang(QQ: 111)"
    assert DoraOpsPlugin._extract_text(msg) == "dict text"
    assert DoraOpsPlugin._mentions(msg)[0].display_name == "yuueang"


def test_plugin_only_treats_at_self_as_bot_mention() -> None:
    msg = SimpleNamespace(self_id="999", message=[FakePlainText("哇 "), FakeAt(111, "yuueang"), FakePlainText(" 大佬")])

    assert DoraOpsPlugin._message_text(msg) == "哇 @yuueang(QQ: 111) 大佬"
    assert DoraOpsPlugin._mentions_bot(msg) is False


def test_plugin_rewrites_raw_cq_at_segments_for_context() -> None:
    msg = SimpleNamespace(
        self_id="3844055063",
        raw_message="[CQ:at,qq=3844055063] 现在可以分辨 [CQ:at,qq=405682409] 吗",
        message=[],
    )

    assert DoraOpsPlugin._message_text(msg) == "@QQ: 3844055063 现在可以分辨 @QQ: 405682409 吗"
    assert DoraOpsPlugin._mentions_bot(msg) is True
    assert [mention.user_id for mention in DoraOpsPlugin._mentions(msg)] == ["3844055063", "405682409"]


@pytest.mark.asyncio
async def test_plugin_resolves_raw_cq_mentions_with_group_member_info() -> None:
    plugin = object.__new__(DoraOpsPlugin)
    qq = FakeQQAPI()
    qq.query.members = {
        "3844055063": SimpleNamespace(card="多萝", nickname="Dora"),
        "405682409": SimpleNamespace(card="", nickname="李瑾"),
    }
    plugin.api = SimpleNamespace(qq=qq)
    msg = SimpleNamespace(
        self_id="3844055063",
        raw_message="[CQ:at,qq=3844055063] 可以分辨 [CQ:at,qq=405682409] 吗",
        message=[],
    )

    text, mentions = await plugin._group_message_text_and_mentions(456, msg)

    assert text == "@多萝(QQ: 3844055063) 可以分辨 @李瑾(QQ: 405682409) 吗"
    assert [(mention.user_id, mention.display_name) for mention in mentions] == [
        ("3844055063", "多萝"),
        ("405682409", "李瑾"),
    ]
    assert qq.query.calls == [(456, "3844055063"), (456, "405682409")]


@pytest.mark.asyncio
async def test_plugin_sends_messages_through_ncatbot_qq_api() -> None:
    plugin = object.__new__(DoraOpsPlugin)
    qq = FakeQQAPI()
    plugin.api = SimpleNamespace(qq=qq)

    await plugin._send_private_reply(123, "私聊")
    await plugin._send_group_reply(456, "群聊", at_user_id=789)

    assert qq.private_messages == [(123, {"text": "私聊"})]
    assert qq.group_messages == [(456, {"text": " 群聊", "at": 789})]


@pytest.mark.asyncio
async def test_plugin_debounces_group_messages_as_buffered_messages() -> None:
    plugin = object.__new__(DoraOpsPlugin)
    runtime = FakeDebounceRuntime()
    plugin.runtime = runtime

    await plugin._enqueue_group_message(group_id=456, user_id=1, nickname="a", text="多萝", mentions_bot=True)
    await plugin._enqueue_group_message(group_id=456, user_id=2, nickname="b", text="补充", mentions_bot=False)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert len(runtime.messages) == 1
    msg = runtime.messages[0]
    assert msg.user_id == 2
    assert msg.mentions_bot is True
    assert [item.text for item in msg.buffered_messages] == ["多萝", "补充"]
    assert [item.user_id for item in msg.buffered_messages] == [1, 2]


@pytest.mark.asyncio
async def test_plugin_sends_group_feedback_notifications_to_admins() -> None:
    plugin = object.__new__(DoraOpsPlugin)
    qq = FakeQQAPI()
    plugin.api = SimpleNamespace(qq=qq)
    plugin.runtime = SimpleNamespace(config=SimpleNamespace(admin=SimpleNamespace(user_ids={456, 123})))

    await plugin._send_admin_notifications("群聊反馈已记录：#1")

    assert qq.private_messages == [
        (123, {"text": "群聊反馈已记录：#1"}),
        (456, {"text": "群聊反馈已记录：#1"}),
    ]


@pytest.mark.asyncio
async def test_plugin_sends_welcome_message_for_group_increase(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)
    plugin = object.__new__(DoraOpsPlugin)
    qq = FakeQQAPI()
    qq.query.members = {"789": SimpleNamespace(card="新朋友", nickname="fallback")}
    plugin.api = SimpleNamespace(qq=qq)
    plugin.runtime = runtime

    await plugin._handle_group_increase(group_id=456, user_id=789, operator_id=123)

    assert qq.query.calls == [(456, "789")]
    assert qq.group_messages == [
        (456, {"text": " 欢迎 新朋友 加入 Dora 社区！\nQQ：789", "at": 789})
    ]


@pytest.mark.asyncio
async def test_plugin_skips_welcome_message_for_disabled_group(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)
    plugin = object.__new__(DoraOpsPlugin)
    qq = FakeQQAPI()
    plugin.api = SimpleNamespace(qq=qq)
    plugin.runtime = runtime

    await plugin._handle_group_increase(group_id=999, user_id=789, operator_id=123)

    assert qq.group_messages == []


@pytest.mark.asyncio
async def test_daily_progress_report_watches_configured_groups() -> None:
    class FakeSummaries:
        async def create_yesterday_progress_jobs(self, jobs):
            assert jobs == "jobs"
            return [("dora_ssr", 1), ("yuescript", 2)]

    plugin = object.__new__(DoraOpsPlugin)
    plugin.runtime = SimpleNamespace(
        summaries=FakeSummaries(),
        jobs="jobs",
        config=SimpleNamespace(
            scheduler=SimpleNamespace(daily_summary_group_ids={456}),
            group_chat=SimpleNamespace(enabled_group_ids={789}),
        ),
    )
    watched = []

    async def fake_watch_progress_jobs(job_ids, send):
        watched.append((job_ids, send))

    plugin._watch_progress_jobs = fake_watch_progress_jobs
    await plugin.daily_progress_report()

    assert watched == [([1, 2], plugin._send_daily_progress_result_to_groups)]


def test_daily_summary_group_ids_fall_back_to_enabled_group_chat() -> None:
    plugin = object.__new__(DoraOpsPlugin)
    plugin.runtime = SimpleNamespace(
        config=SimpleNamespace(
            scheduler=SimpleNamespace(daily_summary_group_ids=set()),
            group_chat=SimpleNamespace(enabled_group_ids={789, 456}),
        )
    )

    assert plugin._daily_summary_group_ids() == [456, 789]


@pytest.mark.asyncio
async def test_daily_progress_group_send_records_chat_history(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)
    runtime.config.scheduler.daily_summary_group_ids.add(456)
    plugin = object.__new__(DoraOpsPlugin)
    qq = FakeQQAPI()
    plugin.api = SimpleNamespace(qq=qq)
    plugin.runtime = runtime

    await plugin._send_daily_progress_result_to_groups("昨日进展分析结果：完成")

    assert qq.group_messages == [(456, {"text": "昨日进展分析结果：完成"})]
    recent = await runtime.storage.list_recent_chat_messages("group:456", 10)
    assert [message["role"] for message in recent] == ["assistant"]
    assert recent[0]["content"] == "昨日进展分析结果：完成"


@pytest.mark.asyncio
async def test_progress_results_send_raw_output_to_llm_summary(tmp_path: Path) -> None:
    output = tmp_path / "output.json"
    output.write_text(
        """```json
{
  "summary": "版本升级至 1.7.7，Agent 系统增强。",
  "commits": ["8381194c2 Refine agent handoff and LLM configuration"],
  "user_visible_changes": ["版本号从 1.7.6 升级到 1.7.7", "Agent 面板新增变更集"],
  "developer_notes": ["修复 DB 查询 double 值截断"],
  "risks": ["需要验证配置兼容性"],
  "recommended_actions": ["发布前跑一轮回归"],
  "announcement": "",
  "should_notify_group": true
}
```""",
        encoding="utf-8",
    )
    fake = FakeChatClient("多萝整理版日报")
    plugin = object.__new__(DoraOpsPlugin)
    plugin.runtime = SimpleNamespace(admin=SimpleNamespace(chat_client=fake), group_chat=SimpleNamespace(chat_client=None))

    result = await plugin._format_progress_results(
        [1],
        {
            1: {
                "id": 1,
                "status": JobStatus.SUCCEEDED.value,
                "tmux_session": "dora_job_123_dora_ssr_progress",
                "output_path": str(output),
            }
        },
    )

    assert result == "多萝整理版日报"
    assert len(fake.calls) == 1
    payload = fake.calls[0][1]["content"]
    assert "版本升级至 1.7.7" in payload
    assert "opencode_output" in payload
    assert "```json" in payload


@pytest.mark.asyncio
async def test_progress_results_fallback_clips_raw_output(tmp_path: Path) -> None:
    output = tmp_path / "output.json"
    output.write_text(
        """```json
{
  "summary": "昨日无提交，仓库无变更。",
  "commits": [],
  "user_visible_changes": [],
  "developer_notes": ["最近一次提交为 2026-04-24"],
  "risks": ["连续 3 天无提交"],
  "recommended_actions": ["确认是否有待合并分支"],
  "announcement": "",
  "should_notify_group": false
}
```""",
        encoding="utf-8",
    )
    plugin = object.__new__(DoraOpsPlugin)
    plugin.runtime = SimpleNamespace(admin=SimpleNamespace(chat_client=None), group_chat=SimpleNamespace(chat_client=None))

    result = await plugin._format_progress_results(
        [1],
        {
            1: {
                "id": 1,
                "status": JobStatus.SUCCEEDED.value,
                "tmux_session": "dora_job_123_yuescript_progress",
                "output_path": str(output),
            }
        },
    )

    assert "昨日无提交" in result
    assert "连续 3 天无提交" in result
    assert "```json" in result
    assert '"summary"' in result


@pytest.mark.asyncio
async def test_feedback_analysis_result_uses_llm_summary(tmp_path: Path) -> None:
    output = tmp_path / "output.json"
    output.write_text('{"summary":"需要补充最小复现","suggested_reply":"请补充平台和日志"}', encoding="utf-8")
    fake = FakeChatClient("多萝整理版反馈分析")
    plugin = object.__new__(DoraOpsPlugin)
    plugin.runtime = SimpleNamespace(admin=SimpleNamespace(chat_client=fake), group_chat=SimpleNamespace(chat_client=None))

    result = await plugin._format_feedback_analysis_result(
        {
            "id": 7,
            "status": JobStatus.SUCCEEDED.value,
            "output_path": str(output),
        }
    )

    assert result == "多萝整理版反馈分析"
    assert len(fake.calls) == 1
    assert "反馈分析结果总结任务" in fake.calls[0][0]["content"]
    assert "需要补充最小复现" in fake.calls[0][1]["content"]


@pytest.mark.asyncio
async def test_feedback_analysis_watcher_replies_to_group_requester(tmp_path: Path) -> None:
    output = tmp_path / "output.json"
    error = tmp_path / "error.log"
    exit_code = tmp_path / "exit_code"
    done = tmp_path / "done"
    prompt = tmp_path / "prompt.md"
    output.write_text('{"summary":"分析完成"}', encoding="utf-8")
    error.write_text("", encoding="utf-8")
    exit_code.write_text("0", encoding="utf-8")
    done.touch()
    prompt.write_text("prompt", encoding="utf-8")
    plugin = object.__new__(DoraOpsPlugin)
    qq = FakeQQAPI()
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)
    plugin.api = SimpleNamespace(qq=qq)
    plugin.runtime = runtime
    job_id = await runtime.storage.create_job(
        kind="feedback_analysis",
        target_type="feedback",
        target_id=1,
        tmux_session="finished_feedback",
        prompt_path=prompt,
        output_path=output,
        error_path=error,
        exit_code_path=exit_code,
        done_path=done,
    )

    await plugin._watch_feedback_analysis_job(job_id, 456, 789)

    assert qq.group_messages == [(456, {"text": " 分析完成：{\"summary\":\"分析完成\"}", "at": 789})]
    recent = await runtime.storage.list_recent_chat_messages("group:456", 10)
    assert [message["role"] for message in recent] == ["assistant"]
    assert recent[0]["content"] == "分析完成：{\"summary\":\"分析完成\"}"
    delivery = await runtime.storage.get_analysis_delivery(job_id)
    assert delivery is None


@pytest.mark.asyncio
async def test_admin_private_chat_records_feedback_and_approval(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)

    ignored = await runtime.handle_admin_text("Dora SSR Web IDE 无法刷新", user_id=999)
    assert ignored is None

    result = await runtime.handle_admin_text("Dora SSR Web IDE 创建文件后无法刷新", user_id=123)
    assert result is not None
    assert "已记录为 #1" in result
    assert "/approve feedback 1" in result
    assert "审批 #1" in result

    feedback = await runtime.storage.get_feedback(1)
    assert feedback is not None
    assert feedback["group_id"] is None
    assert feedback["user_id"] == 123
    assert feedback["project"] == "Dora-SSR"
    approval = await runtime.storage.get_pending_approval("feedback", 1)
    assert approval is not None
    assert approval["requested_group_id"] is None
    assert approval["command"] == "/approve feedback 1"
    messages = await runtime.storage.list_recent_chat_messages("private:123", 10)
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "Dora SSR Web IDE 创建文件后无法刷新"


@pytest.mark.asyncio
async def test_admin_private_chat_fallback_allows_banter(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)

    result = await runtime.handle_admin_text("可以聊聊吗", user_id=123)
    assert result is not None
    assert "可以啊" in result
    assert "游戏引擎" in result


@pytest.mark.asyncio
async def test_admin_private_chat_greeting_falls_back_to_banter_without_llm(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)

    result = await runtime.handle_admin_text("你好", user_id=123)
    assert result is not None
    assert "可以啊" in result
    assert "游戏引擎" in result


@pytest.mark.asyncio
async def test_admin_private_chat_greeting_uses_llm_when_enabled(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(LLM_CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)
    fake = FakeChatClient()
    runtime.admin.chat_client = fake  # type: ignore[assignment]

    result = await runtime.handle_admin_text("你好", user_id=123)

    assert result == "LLM 多轮回复"
    assert len(fake.calls) == 1
    assert fake.calls[0][-1] == {"role": "user", "content": "你好"}


@pytest.mark.asyncio
async def test_admin_private_chat_uses_limited_llm_context(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(LLM_CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)
    fake = FakeChatClient()
    runtime.admin.chat_client = fake  # type: ignore[assignment]

    first = await runtime.handle_admin_text("你好", user_id=123)
    assert first == "LLM 多轮回复"
    second = await runtime.handle_admin_text("继续说明一下", user_id=123)
    assert second == "LLM 多轮回复"

    assert len(fake.calls) == 2
    last_call = fake.calls[-1]
    system_prompt = last_call[0]["content"]
    assert "# 角色设定" in system_prompt
    assert "坏酷又讨人喜欢的小萝莉" in system_prompt
    assert "# 聊天风格限制" in system_prompt
    assert "多萝是在 QQ 群里聊天，不是在写小说、剧本、轻小说或角色扮演小剧场" in system_prompt
    assert "禁止出现类似“（叹气）”“（探头）”“（抬眼）”“（坏笑）”“（小声）”这样的写法" in system_prompt
    assert "/approve feedback <id>" in system_prompt
    assert "/test daily-summary --progress" in system_prompt
    assert "# 能力触发规则" in system_prompt
    assert "应交给仓库分析流程，不要用聊天模型直接回答" in system_prompt
    assert "# 输出长度规则" in system_prompt
    assert "闲聊、问候、调侃、普通接话：只回复一句，尽量不超过 15 个字" in system_prompt
    assert "# 私聊任务规则" in system_prompt
    chat_messages = [message for message in last_call if message["role"] in {"user", "assistant"}]
    assert chat_messages == [
        {"role": "assistant", "content": "LLM 多轮回复"},
        {"role": "user", "content": "继续说明一下"},
    ]


@pytest.mark.asyncio
async def test_admin_private_chat_uses_llm_classifier_for_feedback(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(LLM_CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)
    runtime.admin.chat_client = FakeChatClient("收到，已经记下了。")  # type: ignore[assignment]
    runtime.admin.classifier_client = FakeClassifierClient(
        {
            "should_accept": True,
            "kind": "feedback",
            "project": "Dora-SSR",
            "confidence": 0.91,
            "needs_repo_analysis": True,
            "summary": "启动后黑屏",
        }
    )  # type: ignore[assignment]

    result = await runtime.handle_admin_text("Dora SSR 启动后黑屏", user_id=123)

    assert result == "收到，已经记下了。"
    feedback = await runtime.storage.get_feedback(1)
    assert feedback is not None
    assert feedback["project"] == "Dora-SSR"
    assert feedback["kind"] == "feedback"
    approval = await runtime.storage.get_pending_approval("feedback", 1)
    assert approval is not None


@pytest.mark.asyncio
async def test_admin_private_project_question_records_without_llm_reply(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(LLM_CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)
    chat = FakeChatClient("不应该直接回答")
    runtime.admin.chat_client = chat  # type: ignore[assignment]
    runtime.admin.classifier_client = FakeClassifierClient(
        {
            "should_accept": False,
            "kind": "project_question",
            "project": "Dora-SSR",
            "confidence": 0.91,
            "needs_repo_analysis": False,
            "summary": "询问渲染管线拆分",
        }
    )  # type: ignore[assignment]

    result = await runtime.handle_admin_text("Dora SSR 渲染管线怎么拆比较稳", user_id=123)

    assert result is not None
    assert "已记录为 #1" in result
    assert "/approve feedback 1" in result
    assert chat.calls == []
    feedback = await runtime.storage.get_feedback(1)
    assert feedback is not None
    assert feedback["kind"] == "project_question"
    approval = await runtime.storage.get_pending_approval("feedback", 1)
    assert approval is not None


@pytest.mark.asyncio
async def test_admin_approve_uses_planner_rejection(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(LLM_CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)
    feedback_id = await runtime.storage.create_feedback(
        original_text="这个要不要跑仓库分析？",
        group_id=456,
        user_id=789,
        project="Dora-SSR",
        kind="project_question",
        title="可能需要分析",
        normalized_summary="可能需要分析",
    )
    await runtime.storage.create_approval_request(
        target_type="feedback",
        target_id=feedback_id,
        requested_by=789,
        requested_group_id=456,
        command=f"/approve feedback {feedback_id}",
    )
    runtime.admin.planner_client = FakePlannerClient(
        {
            "should_create_analysis": False,
            "repo_key": None,
            "title": "普通追问",
            "analysis_task": "",
            "context_summary": "上下文显示这不是可分析的仓库问题。",
            "reject_reason": "信息不足，无法形成仓库分析任务。",
            "questions_for_user": ["请补充具体仓库、报错或复现步骤。"],
            "confidence": "high",
        }
    )  # type: ignore[assignment]

    async def fail_create_feedback_analysis(*args, **kwargs):
        raise AssertionError("analysis job should not be created")

    runtime.admin.jobs.create_feedback_analysis = fail_create_feedback_analysis  # type: ignore[method-assign]

    result = await runtime.handle_admin_text(f"/approve feedback {feedback_id}", user_id=123)

    assert result is not None
    assert "不建议创建仓库分析任务" in result
    assert "信息不足" in result
    assert await runtime.storage.get_pending_approval("feedback", feedback_id) is None


@pytest.mark.asyncio
async def test_group_chat_test_command_simulates_group_message(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)

    result = await runtime.handle_admin_text(
        "/test group-chat 456 Dora SSR Web IDE 创建文件后无法刷新",
        user_id=123,
    )

    assert result is not None
    assert "群聊测试结果：" in result
    assert "群：456" in result
    assert "分类：feedback" in result
    assert "项目：Dora-SSR" in result
    assert "记录：1" in result
    assert "审批：1" in result
    assert "回复：收到" not in result
    assert "管理员通知：群聊反馈已记录：#1" in result
    feedback = await runtime.storage.get_feedback(1)
    assert feedback is not None
    assert feedback["group_id"] == 456


@pytest.mark.asyncio
async def test_welcome_test_command_previews_template(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)

    result = await runtime.handle_admin_text(
        "/test welcome 456 789 新朋友",
        user_id=123,
    )

    assert result is not None
    assert "欢迎词测试结果：" in result
    assert "群：456" in result
    assert "用户：新朋友 (789)" in result
    assert "欢迎词：欢迎 新朋友 加入 Dora 社区！\nQQ：789" in result


@pytest.mark.asyncio
async def test_welcome_test_command_reports_disabled_group(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)

    result = await runtime.handle_admin_text(
        "/test welcome 999 789 新朋友",
        user_id=123,
    )

    assert result is not None
    assert "结果：未发送（欢迎词未启用或群未配置）" in result


@pytest.mark.asyncio
async def test_welcome_test_command_requires_group_and_user_id(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)

    result = await runtime.handle_admin_text("/test welcome 456", user_id=123)

    assert result == "格式错误：/test welcome <群号> <QQ号> [昵称]"


@pytest.mark.asyncio
async def test_group_chat_test_requires_group_id_in_private_chat(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)

    private_error = await runtime.handle_admin_text("/test group-chat Dora SSR 报错", user_id=123)
    assert private_error == "格式错误：/test group-chat <群号> <文本>"

    result = await runtime.handle_admin_text("/test group-chat Dora SSR 报错", user_id=123, group_id=789)
    assert result is None


@pytest.mark.asyncio
async def test_repo_check_uses_tracker_status_check(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)

    async def fake_check_repo(repo_key: str, *, is_test: bool = False, triggered_by: str | None = None) -> dict[str, object]:
        assert repo_key == "dora_ssr"
        assert is_test is True
        assert triggered_by == "123"
        return {
            "repo_key": repo_key,
            "head": "abcdef1234567890",
            "old_head": None,
            "changed": False,
            "new_tags": [],
            "change_id": 1,
            "commit_count": 0,
            "changed_files": [],
            "diff_stat": "",
        }

    runtime.tracker.check_repo = fake_check_repo  # type: ignore[method-assign]

    result = await runtime.handle_admin_text("/test repo-check Dora-SSR", user_id=123)
    assert result is not None
    assert "仓库：dora_ssr" in result
    assert "HEAD：abcdef123456" in result
    assert "新增 tag：-" in result


@pytest.mark.asyncio
async def test_job_status_reconciles_and_shows_output_summary(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    prompt = job_dir / "prompt.md"
    output = job_dir / "output.json"
    error = job_dir / "error.log"
    exit_code = job_dir / "exit_code"
    done = job_dir / "done"
    prompt.write_text("prompt", encoding="utf-8")
    output.write_text('{"summary":"昨日没有用户可见变更。"}', encoding="utf-8")
    error.write_text("", encoding="utf-8")
    exit_code.write_text("0", encoding="utf-8")
    done.touch()
    await runtime.storage.create_job(
        kind="daily_progress",
        target_type="repository",
        target_id=None,
        tmux_session="session",
        prompt_path=prompt,
        output_path=output,
        error_path=error,
        exit_code_path=exit_code,
        done_path=done,
        is_test=True,
        triggered_by="123",
        trigger_source="admin_command",
    )

    result = await runtime.handle_admin_text("/test job-status --include-test", user_id=123)
    assert result is not None
    assert "daily_progress succeeded" in result
    assert "结果：昨日没有用户可见变更。" in result


@pytest.mark.asyncio
async def test_job_status_marks_missing_done_finished_tmux_as_failed(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    prompt = job_dir / "prompt.md"
    output = job_dir / "output.json"
    error = job_dir / "error.log"
    exit_code = job_dir / "exit_code"
    done = job_dir / "done"
    prompt.write_text("prompt", encoding="utf-8")
    output.write_text("", encoding="utf-8")
    error.write_text("", encoding="utf-8")
    exit_code.write_text("", encoding="utf-8")
    job_id = await runtime.storage.create_job(
        kind="daily_progress",
        target_type="repository",
        target_id=None,
        tmux_session="missing_session",
        prompt_path=prompt,
        output_path=output,
        error_path=error,
        exit_code_path=exit_code,
        done_path=done,
        is_test=True,
        triggered_by="123",
        trigger_source="admin_command",
    )
    await runtime.storage.update_job_status(job_id, JobStatus.RUNNING)

    async def fake_tmux_session_exists(_session: str) -> bool:
        return False

    runtime.jobs._tmux_session_exists = fake_tmux_session_exists  # type: ignore[method-assign]

    result = await runtime.handle_admin_text("/test job-status --include-test", user_id=123)
    assert result is not None
    assert "daily_progress failed" in result
    assert "tmux session ended before writing job completion marker" in result


@pytest.mark.asyncio
async def test_job_status_shows_five_job_window_with_offset(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)

    for index in range(12):
        job_dir = tmp_path / f"job-{index}"
        job_dir.mkdir()
        prompt = job_dir / "prompt.md"
        output = job_dir / "output.json"
        error = job_dir / "error.log"
        exit_code = job_dir / "exit_code"
        done = job_dir / "done"
        prompt.write_text("prompt", encoding="utf-8")
        output.write_text("", encoding="utf-8")
        error.write_text("", encoding="utf-8")
        exit_code.write_text("", encoding="utf-8")
        await runtime.storage.create_job(
            kind=f"job_{index}",
            target_type="repository",
            target_id=None,
            tmux_session=f"session_{index}",
            prompt_path=prompt,
            output_path=output,
            error_path=error,
            exit_code_path=exit_code,
            done_path=done,
            is_test=True,
            triggered_by="123",
            trigger_source="admin_command",
        )

    result = await runtime.handle_admin_text("/test job-status --include-test", user_id=123)
    assert result is not None
    assert "#12 job_11" in result
    assert "#8 job_7" in result
    assert "#7 job_6" not in result

    result = await runtime.handle_admin_text("/test job-status --include-test --offset 5", user_id=123)
    assert result is not None
    assert "#7 job_6" in result
    assert "#3 job_2" in result
    assert "#8 job_7" not in result
    assert "#2 job_1" not in result
