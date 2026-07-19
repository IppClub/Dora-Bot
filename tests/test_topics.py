import asyncio
from datetime import datetime, time, timedelta
from pathlib import Path
import re
import subprocess
from zoneinfo import ZoneInfo

import pytest

from dora_ops.runtime import DoraOpsRuntime
from dora_ops.topics import TopicArchive


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


class _FakeTopicDesignJobs:
    def __init__(self) -> None:
        self.repo_path: Path | None = None
        self.topic_markdown = ""
        self.prompt = ""

    async def create_topic_design(self, *, repo_key: str, repo_path: Path, ticket_id: int, prompt_text: str) -> int:
        assert repo_key == "dora_ssr"
        assert ticket_id == 1
        self.repo_path = repo_path
        self.prompt = prompt_text
        topic_match = re.search(r"Read the complete ticket discussion from:\n(.+)", prompt_text)
        design_match = re.search(r"Write the final Markdown design document directly to this exact path:\n(.+)", prompt_text)
        assert topic_match is not None
        assert design_match is not None
        self.topic_markdown = Path(topic_match.group(1)).read_text(encoding="utf-8")
        Path(design_match.group(1)).parent.mkdir(parents=True, exist_ok=True)
        Path(design_match.group(1)).write_text("# ds1：OpenCode 分析\n\n## 现有项目分析\n\n- 已检查本地源码。\n", encoding="utf-8")
        return 101

    async def wait_for_job(self, job_id: int) -> dict[str, object]:
        assert job_id == 101
        return {"id": job_id}


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
  design_generator: fallback
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


@pytest.mark.asyncio
async def test_daily_review_uses_opencode_in_project_and_commits_design(tmp_path: Path) -> None:
    remote = tmp_path / "remote.git"
    worktree = tmp_path / "topic-repo"
    source = tmp_path / "Dora-SSR"
    _git("init", "--bare", "--initial-branch=main", str(remote), cwd=tmp_path)
    _git("init", "--initial-branch=main", str(worktree), cwd=tmp_path)
    _git("config", "user.name", "Dora Bot", cwd=worktree)
    _git("config", "user.email", "dora@example.invalid", cwd=worktree)
    _git("remote", "add", "origin", str(remote), cwd=worktree)
    _git("init", "--initial-branch=main", str(source), cwd=tmp_path)

    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(
        f"""
repositories:
  dora_ssr:
    name: Dora SSR
    remote: https://example.invalid/Dora-SSR.git
    local_path: {source}
topic_archive:
  enabled: true
  design_generator: opencode
  repository:
    local_path: {worktree}
    remote: {remote}
    branch: main
""",
        encoding="utf-8",
    )
    runtime = await DoraOpsRuntime.create(config_path)
    fake_jobs = _FakeTopicDesignJobs()
    runtime.topic_reviews.jobs = fake_jobs  # type: ignore[assignment]
    ticket = await runtime.topics.create_ticket(
        group_id=456, user_id=789, nickname="tester", project="Dora-SSR", kind="feature",
        title="OpenCode 日审", original_text="需要在项目中分析后形成设计稿", messages=(),
    )
    yesterday = datetime.combine(datetime.now(ZoneInfo("Asia/Shanghai")).date() - timedelta(days=1), time(hour=12), tzinfo=ZoneInfo("Asia/Shanghai"))
    await runtime.storage.append_topic_message(
        ticket_id=int(ticket["id"]), group_id=456, user_id=789, nickname="tester",
        content="ds1 请先分析 Dora-SSR 的现有实现。", source_message_at=int(yesterday.timestamp()),
    )

    assert await runtime.topic_reviews.review_yesterday() == ["ds1"]
    assert fake_jobs.repo_path == source
    assert "# ds1：OpenCode 日审" in fake_jobs.topic_markdown
    assert "完整讨论线程" in fake_jobs.topic_markdown
    assert str(worktree / "topics" / "ds1.md") in fake_jobs.prompt
    assert (worktree / "designs" / "ds1.md").read_text(encoding="utf-8").startswith("# ds1：OpenCode 分析")
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


def test_design_url_uses_public_archive_location(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(
        """
repositories:
  dora_ssr:
    name: Dora SSR
    remote: https://example.invalid/Dora-SSR.git
topic_archive:
  public_url: https://atomgit.com/ippclub/dora-topic/
  repository:
    branch: main
""",
        encoding="utf-8",
    )
    runtime = asyncio.run(DoraOpsRuntime.create(config_path))
    assert runtime.topic_reviews.design_url("DS12") == "https://atomgit.com/ippclub/dora-topic/blob/main/designs/ds12.md"


@pytest.mark.asyncio
async def test_failed_opencode_cleanup_removes_only_untracked_topic(tmp_path: Path) -> None:
    worktree = tmp_path / "topic-repo"
    _git("init", "--initial-branch=main", str(worktree), cwd=tmp_path)
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(
        f"""
repositories:
  dora_ssr:
    name: Dora SSR
    remote: https://example.invalid/Dora-SSR.git
topic_archive:
  repository:
    local_path: {worktree}
    branch: main
""",
        encoding="utf-8",
    )
    runtime = await DoraOpsRuntime.create(config_path)
    archive = TopicArchive(tmp_path, runtime.config)
    topic_path = Path("topics/ds6.md")
    (worktree / topic_path).parent.mkdir(parents=True)
    (worktree / topic_path).write_text("temporary topic", encoding="utf-8")

    await archive.discard_unwritten_topic(topic_path)

    assert not (worktree / topic_path).exists()
