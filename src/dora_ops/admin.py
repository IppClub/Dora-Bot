from __future__ import annotations

import json
import logging
from pathlib import Path

from .classifier import classify_text, classify_text_with_llm
from .config import BotConfig
from .group_chat import DORA_PERSONA_PROMPT, GroupMessageInput, GroupMessageResult, GroupMessageService
from .jobs import JobManager
from .llm import LLMError, OpenAICompatibleChatClient
from .prompts import feedback_analysis_prompt, repo_diff_prompt
from .repo_tracker import RepoTracker
from .storage import Storage
from .summary import SummaryService


try:
    from ncatbot.utils import get_log
except Exception:  # pragma: no cover - lets core tests run without NcatBot internals.
    get_log = None  # type: ignore[assignment]


REPO_ALIASES = {
    "dora-ssr": "dora_ssr",
    "dora_ssr": "dora_ssr",
    "dora": "dora_ssr",
    "yuescript": "yuescript",
    "yue": "yuescript",
}
logger = get_log("DoraOps.Admin") if get_log is not None else logging.getLogger(__name__)
JOB_STATUS_WINDOW = 5


class AdminCommands:
    def __init__(
        self,
        base_dir: Path,
        config: BotConfig,
        storage: Storage,
        tracker: RepoTracker,
        jobs: JobManager,
        summaries: SummaryService,
        group_chat: GroupMessageService,
        chat_client: OpenAICompatibleChatClient | None = None,
        classifier_client: OpenAICompatibleChatClient | None = None,
    ):
        self.base_dir = base_dir
        self.config = config
        self.storage = storage
        self.tracker = tracker
        self.jobs = jobs
        self.summaries = summaries
        self.group_chat = group_chat
        self.chat_client = chat_client
        self.classifier_client = classifier_client

    def is_admin(self, user_id: int, group_id: int | None = None) -> bool:
        if user_id in self.config.admin.user_ids:
            return True
        return group_id is not None and group_id in self.config.admin.group_ids

    async def handle(self, text: str, *, user_id: int, group_id: int | None = None) -> str | None:
        if group_id is not None:
            logger.info("admin command skipped: group context user=%s group=%s", user_id, group_id)
            return None
        if not (text.startswith("/test") or text.startswith("/approve") or text.startswith("/reject") or text.startswith("/approvals")):
            if not self.is_admin(user_id):
                logger.info("admin private chat skipped: non-admin user=%s", user_id)
                return None
            return await self._handle_private_chat(text, user_id=user_id)
        if not self.is_admin(user_id):
            logger.info("admin command denied: user=%s text=%r", user_id, self._clip_log(text))
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
        logger.info("admin command accepted: user=%s command=%s arg=%r", user_id, command, self._clip_log(arg))

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

        if command == "group-chat":
            try:
                return await self._handle_group_chat_test(arg, user_id=user_id)
            except ValueError as exc:
                return str(exc)

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
            try:
                offset = self._parse_job_status_offset(arg)
            except ValueError as exc:
                return str(exc)
            jobs = await self.storage.list_recent_jobs(
                limit=JOB_STATUS_WINDOW,
                include_test=include_test,
                offset=offset,
            )
            for job in jobs:
                await self.jobs.reconcile_job(job)
            jobs = await self.storage.list_recent_jobs(
                limit=JOB_STATUS_WINDOW,
                include_test=include_test,
                offset=offset,
            )
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
                "/test group-chat <群号> <文本>",
                "/test repo-check Dora-SSR|YueScript",
                "/test tmux",
                "/test opencode Dora-SSR|YueScript",
                "/test daily-summary --dry-run",
                "/test daily-summary --progress",
                "/test job-status --include-test [--offset 5]",
                "/approvals",
                "/approve feedback <id>",
                "/reject feedback <id>",
            ]
        )

    async def _handle_group_chat_test(self, arg: str, *, user_id: int) -> str:
        target_group_id, text = self._parse_group_chat_test(arg)
        result = await self.group_chat_test(target_group_id, user_id=user_id, text=text)
        if result is None:
            return f"群聊测试：无响应\n群：{target_group_id}"
        return self._group_chat_test_text(target_group_id, result)

    async def _handle_private_chat(self, text: str, *, user_id: int) -> str | None:
        normalized = text.strip()
        if not normalized:
            logger.info("admin private chat skipped: empty text user=%s", user_id)
            return None
        conversation_key = self._private_conversation_key(user_id)
        await self.storage.append_chat_message(conversation_key, "user", normalized)
        classification = await self._classify_private_text(normalized)
        logger.info(
            "admin private chat classified: user=%s kind=%s project=%s accept=%s repo_analysis=%s confidence=%.2f",
            user_id,
            classification.kind,
            classification.project,
            classification.should_accept,
            classification.needs_repo_analysis,
            classification.confidence,
        )
        side_effect_note = await self._record_private_chat_feedback(
            normalized,
            user_id=user_id,
            classification=classification,
        )
        if side_effect_note is not None and classification.kind == "project_question":
            reply = self._fallback_private_chat_reply(classification, side_effect_note)
            await self.storage.append_chat_message(conversation_key, "assistant", reply)
            logger.info("admin private project question recorded without llm reply: user=%s reply_len=%s", user_id, len(reply))
            return reply
        if self.config.llm.enabled and self.chat_client is not None:
            try:
                reply = await self._llm_private_chat_reply(
                    conversation_key,
                    side_effect_note=side_effect_note,
                )
            except LLMError as exc:
                logger.info("admin private chat llm failed: user=%s error=%s", user_id, exc)
                reply = f"LLM 回复失败：{exc}\n{self._fallback_private_chat_reply(classification, side_effect_note)}"
            await self.storage.append_chat_message(conversation_key, "assistant", reply)
            logger.info("admin private chat replied via llm: user=%s reply_len=%s", user_id, len(reply))
            return reply

        reply = self._fallback_private_chat_reply(classification, side_effect_note)
        await self.storage.append_chat_message(conversation_key, "assistant", reply)
        logger.info("admin private chat replied via fallback: user=%s reply_len=%s", user_id, len(reply))
        return reply

    async def _record_private_chat_feedback(self, text: str, *, user_id: int, classification) -> str | None:
        if not classification.should_accept:
            return None

        feedback_id = await self.storage.create_feedback(
            original_text=text,
            group_id=None,
            user_id=user_id,
            project=classification.project,
            kind=classification.kind,
            title=classification.summary[:40],
            normalized_summary=classification.summary,
        )
        if not classification.needs_repo_analysis:
            logger.info("admin private feedback recorded: user=%s feedback=%s repo_analysis=false", user_id, feedback_id)
            return f"已记录为 #{feedback_id}。"

        existing = await self.storage.get_pending_approval("feedback", feedback_id)
        approval_id = int(existing["id"]) if existing is not None else await self.storage.create_approval_request(
            target_type="feedback",
            target_id=feedback_id,
            requested_by=user_id,
            requested_group_id=None,
            command=f"/approve feedback {feedback_id}",
        )
        logger.info("admin private feedback recorded: user=%s feedback=%s approval=%s repo_analysis=true", user_id, feedback_id, approval_id)
        return f"已记录为 #{feedback_id}，可发送 /approve feedback {feedback_id} 批准深度分析。审批 #{approval_id}。"

    @staticmethod
    def _clip_log(text: str, limit: int = 160) -> str:
        compact = " ".join(str(text).split())
        return compact if len(compact) <= limit else f"{compact[:limit].rstrip()}..."

    async def _classify_private_text(self, text: str):
        client = self.classifier_client if self.config.llm.enabled else None
        return await classify_text_with_llm(text, client)

    def _fallback_private_chat_reply(self, classification, side_effect_note: str | None) -> str:
        if not classification.should_accept:
            if classification.project:
                return "看起来和项目有关，但信息还不够。请补充报错全文、平台、版本和最小复现。"
            return "可以啊。想聊游戏引擎、开源，还是单纯摸鱼？先说好，太无聊的话我会嫌弃两句。"

        if not classification.needs_repo_analysis:
            return f"收到，{side_effect_note or '已记录。'}"

        return f"收到，这个像是 {classification.project or '项目'} 的有效问题，{side_effect_note or '可批准深度分析。'}"

    async def _llm_private_chat_reply(self, conversation_key: str, *, side_effect_note: str | None) -> str:
        assert self.chat_client is not None
        recent = await self.storage.list_recent_chat_messages(
            conversation_key,
            self.config.llm.max_context_messages,
        )
        messages = [
            {
                "role": "system",
                "content": (
                    f"{DORA_PERSONA_PROMPT}\n\n"
                    "# 私聊任务规则\n"
                    "- 你正在和管理员私聊，继续使用多萝人格，但优先清楚处理维护事务。\n"
                    "- 使用简洁中文回复管理员。\n"
                    "- 可以帮助记录反馈、说明 /approvals、/approve feedback <id>、/reject feedback <id> 等审批命令。\n"
                    "- 可以说明 /test ping、/test group-chat、/test opencode、/test daily-summary --progress、/test job-status --include-test 等测试命令的用途。\n"
                    "- 不要声称已经执行未执行的操作。"
                ),
            }
        ]
        if side_effect_note:
            messages.append({"role": "system", "content": f"本轮系统记录：{side_effect_note}"})
        messages.extend({"role": str(row["role"]), "content": str(row["content"])} for row in recent)
        return await self.chat_client.complete(messages)

    @staticmethod
    def _private_conversation_key(user_id: int) -> str:
        return f"private:{user_id}"

    @staticmethod
    def _is_greeting(text: str) -> bool:
        normalized = text.strip().lower()
        return normalized in {"你好", "您好", "hi", "hello", "hey", "哈喽", "嗨"}

    async def group_chat_test(self, group_id: int, *, user_id: int, text: str) -> GroupMessageResult | None:
        return await self.group_chat.handle(
            GroupMessageInput(
                group_id=group_id,
                user_id=user_id,
                nickname="admin-test",
                text=text,
                mentions_bot=True,
            )
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
    def _parse_group_chat_test(arg: str) -> tuple[int, str]:
        parts = arg.split(maxsplit=1)
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].strip():
            raise ValueError("格式错误：/test group-chat <群号> <文本>")
        return int(parts[0]), parts[1].strip()

    @staticmethod
    def _group_chat_test_text(group_id: int, result: GroupMessageResult) -> str:
        lines = [
            "群聊测试结果：",
            f"群：{group_id}",
            f"分类：{result.classification.kind}",
            f"项目：{result.classification.project or '-'}",
            f"记录：{result.feedback_id or '-'}",
            f"审批：{result.approval_id or '-'}",
            f"原因：{result.reason}",
        ]
        if result.reply:
            lines.append(f"回复：{result.reply}")
        if result.admin_notification:
            lines.append(f"管理员通知：{result.admin_notification}")
        return "\n".join(lines)

    @staticmethod
    def _parse_job_status_offset(arg: str) -> int:
        parts = arg.split()
        for index, part in enumerate(parts):
            value = None
            if part == "--offset":
                if index + 1 >= len(parts):
                    raise ValueError("格式错误：/test job-status [--include-test] [--offset <数量>]")
                value = parts[index + 1]
            elif part.startswith("--offset="):
                value = part.removeprefix("--offset=")
            elif part.startswith("offset="):
                value = part.removeprefix("offset=")
            if value is not None:
                if not value.isdigit():
                    raise ValueError("offset 必须是非负整数。")
                return int(value)
        return 0

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
        if feedback.get("group_id") is not None:
            await self.storage.create_analysis_delivery(
                job_id=job_id,
                feedback_id=target_id,
                group_id=int(feedback["group_id"]),
                user_id=int(feedback["user_id"]) if feedback.get("user_id") is not None else None,
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
