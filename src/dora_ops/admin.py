from __future__ import annotations

import json
from pathlib import Path

from .classifier import classify_text
from .config import BotConfig, resolve_path
from .jobs import JobManager
from .prompts import repo_diff_prompt
from .repo_tracker import RepoTracker
from .storage import Storage
from .summary import SummaryService


REPO_ALIASES = {
    "dora-ssr": "dora_ssr",
    "dora_ssr": "dora_ssr",
    "dora": "dora_ssr",
    "yuescript": "yuescript",
    "yue": "yuescript",
}


class AdminCommands:
    def __init__(
        self,
        base_dir: Path,
        config: BotConfig,
        storage: Storage,
        tracker: RepoTracker,
        jobs: JobManager,
        summaries: SummaryService,
    ):
        self.base_dir = base_dir
        self.config = config
        self.storage = storage
        self.tracker = tracker
        self.jobs = jobs
        self.summaries = summaries

    def is_admin(self, user_id: int, group_id: int | None = None) -> bool:
        if user_id in self.config.admin.user_ids:
            return True
        return group_id is not None and group_id in self.config.admin.group_ids

    async def handle(self, text: str, *, user_id: int, group_id: int | None = None) -> str | None:
        if not text.startswith("/test"):
            return None
        if not self.is_admin(user_id, group_id):
            return "没有管理员权限。"

        parts = text.split(maxsplit=2)
        if len(parts) == 1:
            return self.help_text()
        command = parts[1]
        arg = parts[2] if len(parts) > 2 else ""
        triggered_by = str(user_id)

        if command == "ping":
            await self.storage.daily_counts(0, include_test=True)
            return "pong：配置已加载，数据库可读写。"

        if command == "classify":
            result = classify_text(arg)
            return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)

        if command == "feedback":
            result = classify_text(arg)
            feedback_id = await self.storage.create_feedback(
                original_text=arg,
                group_id=group_id,
                user_id=user_id,
                project=result.project,
                kind=result.kind,
                title=result.summary[:40],
                normalized_summary=result.summary,
                is_test=True,
            )
            return f"测试反馈已写入：#{feedback_id}\n{json.dumps(result.to_dict(), ensure_ascii=False)}"

        if command == "repo-check":
            repo_key = self._repo_key(arg)
            result = await self.tracker.check_repo(repo_key, is_test=True, triggered_by=triggered_by)
            return self._repo_check_text(result)

        if command == "tmux":
            job_id = await self.jobs.create_tmux_echo_test(triggered_by=triggered_by)
            return f"tmux 测试任务已创建：#{job_id}"

        if command == "opencode":
            repo_key = self._repo_key(arg)
            repo = self.config.repositories[repo_key]
            mirror = await self.tracker.ensure_mirror(repo_key, repo)
            prompt = repo_diff_prompt(
                repo_name=repo.name,
                old_head=None,
                new_head="HEAD",
                commit_count=0,
                changed_files=[],
                diff_stat="Admin smoke test. Read repository context and return a short JSON summary.",
            )
            job_id = await self.jobs.create_opencode_repo_analysis(
                repo_key,
                mirror,
                None,
                prompt,
                is_test=True,
                triggered_by=triggered_by,
            )
            return f"opencode 测试任务已创建：#{job_id}"

        if command == "daily-summary":
            dry_run = "--dry-run" in arg
            return await self.summaries.build_daily_summary(dry_run=dry_run, include_test=True)

        if command == "job-status":
            include_test = "--include-test" in arg
            jobs = await self.storage.list_recent_jobs(include_test=include_test)
            if not jobs:
                return "没有任务记录。"
            return "\n".join(
                f"#{job['id']} {job['kind']} {job['status']} session={job['tmux_session'] or '-'}"
                for job in jobs
            )

        if command == "quota":
            return "当前版本仅实现管理员测试链路，正式群聊额度将在消息处理模块启用。"

        return self.help_text()

    @staticmethod
    def help_text() -> str:
        return "\n".join(
            [
                "可用测试命令：",
                "/test ping",
                "/test classify <文本>",
                "/test feedback <文本>",
                "/test repo-check Dora-SSR|YueScript",
                "/test tmux",
                "/test opencode Dora-SSR|YueScript",
                "/test daily-summary --dry-run",
                "/test job-status --include-test",
            ]
        )

    @staticmethod
    def _repo_key(value: str) -> str:
        key = REPO_ALIASES.get(value.strip().lower())
        if not key:
            raise ValueError(f"未知仓库：{value}")
        return key

    @staticmethod
    def _repo_check_text(result: dict[str, object]) -> str:
        changed_files = result.get("changed_files") or []
        return "\n".join(
            [
                f"仓库：{result['repo_key']}",
                f"HEAD：{str(result['head'])[:12]}",
                f"上次记录：{str(result.get('old_head') or '-')[:12]}",
                f"是否变化：{result['changed']}",
                f"新增 tag：{', '.join(result.get('new_tags') or []) or '-'}",
                f"变更记录：{result.get('change_id') or '-'}",
                f"文件数：{len(changed_files)}",
            ]
        )
