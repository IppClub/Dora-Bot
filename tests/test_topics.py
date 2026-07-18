from datetime import datetime, time, timedelta
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

import pytest

from dora_ops.runtime import DoraOpsRuntime


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


@pytest.mark.asyncio
async def test_daily_review_pushes_design_once_for_unchanged_discussion(tmp_path: Path) -> None:
    remote = tmp_path / "remote.git"
    worktree = tmp_path / "topic-repo"
    _git("init", "--bare", "--initial-branch=main", str(remote), cwd=tmp_path)
    _git("init", "--initial-branch=main", str(worktree), cwd=tmp_path)
    _git("config", "user.name", "Dora Bot", cwd=worktree)
    _git("config", "user.email", "dora@example.invalid", cwd=worktree)
    _git("remote", "add", "origin", str(remote), cwd=worktree)

    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(
        f"""
repositories:
  dora_ssr:
    name: Dora SSR
    remote: https://example.invalid/Dora-SSR.git
    local_path: {tmp_path}
topic_archive:
  enabled: true
  repository:
    local_path: {worktree}
    remote: {remote}
    branch: main
""",
        encoding="utf-8",
    )
    runtime = await DoraOpsRuntime.create(config_path)
    ticket = await runtime.topics.create_ticket(
        group_id=456, user_id=789, nickname="tester", project="Dora-SSR", kind="feature",
        title="每日归档", original_text="需要每日归档讨论", messages=(),
    )
    yesterday = datetime.combine(datetime.now(ZoneInfo("Asia/Shanghai")).date() - timedelta(days=1), time(hour=12), tzinfo=ZoneInfo("Asia/Shanghai"))
    await runtime.storage.append_topic_message(
        ticket_id=int(ticket["id"]), group_id=456, user_id=789, nickname="tester",
        content="确认文稿要记录最后讨论日期。", source_message_at=int(yesterday.timestamp()),
    )

    first = await runtime.topic_reviews.review_yesterday()
    second = await runtime.topic_reviews.review_yesterday()
    assert first == ["ds1"]
    assert second == []
    assert (worktree / "topics" / "ds1.md").exists()
    design = (worktree / "designs" / "ds1.md").read_text(encoding="utf-8")
    assert "讨论更新至" in design
    assert _git_log_count(worktree) == 1


def _git_log_count(worktree: Path) -> int:
    result = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=worktree, check=True, capture_output=True, text=True)
    return int(result.stdout.strip())


@pytest.mark.asyncio
async def test_topic_message_limit_returns_latest_messages_in_order(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(
        """
repositories:
  dora_ssr:
    name: Dora SSR
    remote: https://example.invalid/Dora-SSR.git
    local_path: .
""",
        encoding="utf-8",
    )
    runtime = await DoraOpsRuntime.create(config_path)
    ticket = await runtime.storage.create_topic_ticket(
        group_id=1, user_id=1, project=None, kind="feature", title="limit", original_text="limit",
    )
    for index in range(45):
        await runtime.storage.append_topic_message(
            ticket_id=int(ticket["id"]), group_id=1, user_id=1, nickname="tester", content=f"message {index}",
        )
    latest = await runtime.storage.list_topic_messages(int(ticket["id"]), limit=40)
    assert [row["content"] for row in latest] == [f"message {index}" for index in range(5, 45)]
