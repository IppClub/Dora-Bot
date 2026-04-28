from __future__ import annotations

import json
from pathlib import Path

from .classifier import classify_text
from .config import BotConfig
from .jobs import JobManager
from .prompts import feedback_analysis_prompt, repo_diff_prompt
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
        if group_id is not None:
            return None
        if not (text.startswith("/test") or text.startswith("/approve") or text.startswith("/reject") or text.startswith("/approvals")):
            return None
        if not self.is_admin(user_id):
            return "没有管理员权限。"

        if text.startswith("/approve"):
            try:
                return await self._handle_approve(text, user_id=user_id)
            except ValueError as exc:
                return str(exc)
        if text.startswith("/reject"):
            try:
                return await self._handle_reject(text, user_id=user_id)
            except ValueError as exc:
                return str(exc)
        if text.startswith("/approvals"):
            return await self._handle_approvals()

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
            if "--progress" in arg:
                try:
                    created = await self.summaries.create_yesterday_progress_jobs(
                        self.jobs,
                        triggered_by=triggered_by,
                        is_test=True,
                    )
                except (FileNotFoundError, ValueError) as exc:
                    return f"创建昨日进展分析失败：{exc}"
                lines = ["昨日进展分析任务已创建："]
                lines.extend(f"- {repo_key}: job #{job_id}" for repo_key, job_id in created)
                return "\n".join(lines)
            return await self.summaries.build_daily_summary(dry_run=dry_run, include_test=True)

        if command == "job-status":
            include_test = "--include-test" in arg
            jobs = await self.storage.list_recent_jobs(include_test=include_test)
            for job in jobs:
                await self.jobs.reconcile_job(job)
            jobs = await self.storage.list_recent_jobs(include_test=include_test)
            if not jobs:
                return "没有任务记录。"
            return "\n".join(self._job_status_text(job) for job in jobs)

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
                "/test daily-summary --progress",
                "/test job-status --include-test",
                "/approvals",
                "/approve feedback <id>",
                "/reject feedback <id>",
            ]
        )

    async def _handle_approvals(self) -> str:
        approvals = await self.storage.list_pending_approvals()
        if not approvals:
            return "没有待审批任务。"
        lines = ["待审批任务："]
        for approval in approvals:
            lines.append(
                f"#{approval['id']} {approval['target_type']}:{approval['target_id']} "
                f"群={approval['requested_group_id'] or '-'} 命令：{approval['command']}"
            )
        return "\n".join(lines)

    @staticmethod
    def _job_status_text(job: dict[str, object]) -> str:
        line = f"#{job['id']} {job['kind']} {job['status']} session={job['tmux_session'] or '-'}"
        status = str(job.get("status") or "")
        if status == "succeeded":
            analysis = JobManager._read_analysis(Path(str(job["output_path"])))
            summary = analysis.get("summary") or analysis.get("announcement") or analysis.get("raw")
            if summary:
                line = f"{line}\n结果：{str(summary)[:500]}"
        elif status in {"failed", "timeout"} and job.get("error"):
            line = f"{line}\n错误：{str(job['error'])[:500]}"
        return line

    async def _handle_reject(self, text: str, *, user_id: int) -> str:
        target_type, target_id = self._parse_decision_command(text, "/reject")
        approval = await self.storage.get_pending_approval(target_type, target_id)
        if approval is None:
            return f"没有待拒绝任务：{target_type} {target_id}"
        await self.storage.decide_approval(int(approval["id"]), "rejected", decided_by=user_id)
        return f"已拒绝：{target_type} {target_id}"

    async def _handle_approve(self, text: str, *, user_id: int) -> str:
        target_type, target_id = self._parse_decision_command(text, "/approve")
        approval = await self.storage.get_pending_approval(target_type, target_id)
        if approval is None:
            return f"没有待批准任务：{target_type} {target_id}"
        if target_type != "feedback":
            return f"暂不支持批准类型：{target_type}"

        feedback = await self.storage.get_feedback(target_id)
        if feedback is None:
            return f"反馈不存在：#{target_id}"

        repo_key = self._repo_key_for_project(feedback.get("project"))
        repo = self.config.repositories[repo_key]
        mirror = await self.tracker.ensure_mirror(repo_key, repo)
        prompt = feedback_analysis_prompt(
            repo_name=repo.name,
            project=feedback.get("project"),
            kind=feedback.get("kind"),
            title=feedback.get("title"),
            original_text=str(feedback.get("original_text") or ""),
        )
        job_id = await self.jobs.create_feedback_analysis(
            repo_key,
            mirror,
            target_id,
            prompt,
            triggered_by=str(user_id),
        )
        await self.storage.decide_approval(int(approval["id"]), "approved", decided_by=user_id, note=f"job:{job_id}")
        return f"已批准反馈 #{target_id}，分析任务已创建：#{job_id}"

    @staticmethod
    def _parse_decision_command(text: str, prefix: str) -> tuple[str, int]:
        parts = text.split()
        if len(parts) != 3 or parts[0] != prefix:
            raise ValueError(f"格式错误：{prefix} feedback <id>")
        return parts[1], int(parts[2])

    @staticmethod
    def _repo_key_for_project(project: object) -> str:
        text = str(project or "").lower()
        if "yue" in text:
            return "yuescript"
        return "dora_ssr"

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
