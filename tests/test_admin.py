from pathlib import Path

import pytest

from dora_ops.models import JobStatus
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
    assert "回复：收到" in result
    feedback = await runtime.storage.get_feedback(1)
    assert feedback is not None
    assert feedback["group_id"] == 456


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
