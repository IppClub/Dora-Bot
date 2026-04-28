from pathlib import Path

import pytest

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


@pytest.mark.asyncio
async def test_admin_progress_report_requires_local_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)

    result = await runtime.handle_admin_text("/test daily-summary --progress", user_id=123)
    assert result is not None
    assert "缺少" in result


@pytest.mark.asyncio
async def test_repo_check_creates_recent_commit_analysis_job(tmp_path: Path) -> None:
    repo_path = tmp_path / "Dora-SSR"
    repo_path.mkdir()
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(
        CONFIG.replace(
            "watch_tags: true\n    watch_paths: []",
            f"local_path: {repo_path}\n    watch_tags: true\n    watch_paths: []",
            1,
        ),
        encoding="utf-8",
    )
    runtime = await DoraOpsRuntime.create(config_path)

    async def fake_start_tmux_job(*_args, **_kwargs) -> None:
        return None

    runtime.jobs._start_tmux_job = fake_start_tmux_job  # type: ignore[method-assign]

    result = await runtime.handle_admin_text("/test repo-check Dora-SSR", user_id=123)
    assert result is not None
    assert "最近 24 小时仓库分析任务已创建：#1" in result

    jobs = await runtime.storage.list_recent_jobs(include_test=True)
    assert jobs[0]["kind"] == "recent_commits"
    assert jobs[0]["target_type"] == "repository"
    prompt = Path(jobs[0]["prompt_path"]).read_text(encoding="utf-8")
    assert "Analyze commits from the last 24 hours" in prompt
    assert "git log --since='24 hours ago'" in prompt


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
    output.write_text('{"summary":"最近 24 小时没有用户可见变更。"}', encoding="utf-8")
    error.write_text("", encoding="utf-8")
    exit_code.write_text("0", encoding="utf-8")
    done.touch()
    await runtime.storage.create_job(
        kind="recent_commits",
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
    assert "recent_commits succeeded" in result
    assert "结果：最近 24 小时没有用户可见变更。" in result
