from pathlib import Path

import pytest

from dora_ops.group_chat import GroupMessageInput
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
async def test_group_ignores_test_commands(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)

    result = await runtime.handle_group_message(
        GroupMessageInput(group_id=456, user_id=123, nickname="admin", text="/test ping")
    )
    assert result is None
