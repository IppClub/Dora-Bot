from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
import re
from typing import Awaitable, Callable

try:
    from ncatbot.plugin import NcatBotPlugin
    from ncatbot.core import registrar
    from ncatbot.utils import get_log
except Exception:  # pragma: no cover - lets core tests run without NcatBot internals.
    NcatBotPlugin = object  # type: ignore[misc,assignment]
    registrar = None  # type: ignore[assignment]
    get_log = None  # type: ignore[assignment]

from dora_ops.runtime import DoraOpsRuntime
from dora_ops.group_chat import DORA_PERSONA_PROMPT, GroupBufferedMessage, GroupMessageInput
from dora_ops.llm import LLMError
from dora_ops.models import JobStatus


PROGRESS_JOB_PATTERN = re.compile(r"job #(\d+)")
PROGRESS_SESSION_PATTERN = re.compile(r"dora_job_\d+_(.+)_progress$")
APPROVED_FEEDBACK_JOB_PATTERN = re.compile(r"分析任务已创建：#(\d+)")
logger = get_log("DoraOps") if get_log is not None else logging.getLogger(__name__)


class PendingGroupMessage:
    def __init__(self, *, group_id: int):
        self.group_id = group_id
        self.messages: list[GroupBufferedMessage] = []
        self.mentions_bot = False
        self.task: asyncio.Task | None = None


class DoraOpsPlugin(NcatBotPlugin):  # type: ignore[misc,valid-type]
    name = "dora_ops"
    version = "0.1.0"

    async def on_load(self) -> None:
        self._pending_group_messages: dict[int, PendingGroupMessage] = {}
        self.runtime = await DoraOpsRuntime.create(Path("dora-bot.yaml"))
        logger.info(
            "dora_ops loaded: admins=%s admin_groups=%s group_enabled=%s group_ids=%s llm_enabled=%s",
            sorted(self.runtime.config.admin.user_ids),
            sorted(self.runtime.config.admin.group_ids),
            self.runtime.config.group_chat.enabled,
            sorted(self.runtime.config.group_chat.enabled_group_ids),
            self.runtime.config.llm.enabled,
        )
        if hasattr(self, "add_scheduled_task"):
            self.add_scheduled_task(
                "daily_progress_report",
                self.runtime.config.scheduler.daily_summary_time,
                callback=self.daily_progress_report,
            )
        asyncio.create_task(self._recover_feedback_analysis_jobs())

    async def _recover_feedback_analysis_jobs(self) -> None:
        queued = await self.runtime.storage.list_jobs_by_status(
            JobStatus.QUEUED,
            kind="feedback_analysis",
        )
        for job in queued:
            repo_key = self._repo_key_from_feedback_session(str(job.get("tmux_session") or ""))
            if repo_key is None:
                logger.warning("skip queued feedback job with unrecognized session: job=%s session=%s", job.get("id"), job.get("tmux_session"))
                continue
            repo = self.runtime.config.repositories.get(repo_key)
            if repo is None:
                logger.warning("skip queued feedback job with unknown repo: job=%s repo=%s", job.get("id"), repo_key)
                continue
            try:
                mirror = await self.runtime.tracker.ensure_mirror(repo_key, repo)
                await self.runtime.jobs.resume_queued_feedback_analysis(job, mirror)
                logger.info("recovered queued feedback analysis job: job=%s repo=%s", job.get("id"), repo_key)
            except Exception:
                logger.exception("failed to recover queued feedback analysis job: job=%s", job.get("id"))
        deliveries = await self.runtime.storage.list_pending_analysis_deliveries()
        for delivery in deliveries:
            asyncio.create_task(self._watch_deliverable_feedback_analysis_job(int(delivery["job_id"])))

    @staticmethod
    def _repo_key_from_feedback_session(session: str) -> str | None:
        prefix = "dora_job_"
        marker = "_feedback_"
        if not session.startswith(prefix):
            return None
        rest = session[len(prefix) :]
        _, sep, tail = rest.partition("_")
        if not sep:
            return None
        repo_key, sep, feedback_id = tail.partition(marker)
        if not sep or not repo_key or not feedback_id.isdigit():
            return None
        return repo_key

    async def daily_progress_report(self) -> None:
        try:
            created = await self.runtime.summaries.create_yesterday_progress_jobs(self.runtime.jobs)
        except Exception:
            logger.exception("daily progress report failed to create jobs")
            return
        job_ids = [job_id for _, job_id in created]
        group_ids = self._daily_summary_group_ids()
        logger.info("daily progress report jobs created: jobs=%s groups=%s", job_ids, group_ids)
        if not job_ids:
            logger.info("daily progress report skipped: no jobs created")
            return
        if not group_ids:
            logger.info("daily progress report skipped: no target groups configured")
            return
        await self._watch_progress_jobs(job_ids, self._send_daily_progress_result_to_groups)

    if registrar is not None:

        @registrar.qq.on_private_message()
        async def on_private_message(self, event) -> None:
            text = self._message_text(event)
            user_id = int(getattr(event, "user_id", 0))
            logger.info("private message received: user=%s text=%r", user_id, self._clip_log(text))
            result = await self.runtime.handle_admin_text(text, user_id=user_id)
            logger.info("private message handled: user=%s replied=%s", user_id, bool(result))
            if result:
                await event.reply(text=result)
                self._maybe_watch_daily_progress(text, result, lambda message: event.reply(text=message))
                self._maybe_watch_feedback_analysis(result)

        @registrar.qq.on_group_message()
        async def on_group_message(self, event) -> None:
            text = self._message_text(event)
            user_id = int(getattr(getattr(event, "sender", None), "user_id", 0) or getattr(event, "user_id", 0))
            group_id = int(getattr(event, "group_id", 0))
            mentions_bot = self._mentions_bot(event)
            logger.info(
                "group message received: group=%s user=%s mentions_bot=%s text=%r",
                group_id,
                user_id,
                mentions_bot,
                self._clip_log(text),
            )
            await self._enqueue_group_message(
                group_id=group_id,
                user_id=user_id,
                nickname=str(getattr(getattr(event, "sender", None), "nickname", "")),
                text=text,
                mentions_bot=mentions_bot,
            )

    else:

        async def on_private_message(self, msg) -> None:
            text = self._message_text(msg)
            user_id = int(getattr(msg, "user_id", 0))
            logger.info("private message received: user=%s text=%r", user_id, self._clip_log(text))
            result = await self.runtime.handle_admin_text(text, user_id=user_id)
            logger.info("private message handled: user=%s replied=%s", user_id, bool(result))
            if result:
                await self._send_private_reply(user_id, result)
                self._maybe_watch_daily_progress(
                    text,
                    result,
                    lambda message: self._send_private_reply(user_id, message),
                )
                self._maybe_watch_feedback_analysis(result)

        async def on_group_message(self, msg) -> None:
            text = self._message_text(msg)
            user_id = int(getattr(getattr(msg, "sender", None), "user_id", 0) or getattr(msg, "user_id", 0))
            group_id = int(getattr(msg, "group_id", 0))
            mentions_bot = self._mentions_bot(msg)
            logger.info(
                "group message received: group=%s user=%s mentions_bot=%s text=%r",
                group_id,
                user_id,
                mentions_bot,
                self._clip_log(text),
            )
            await self._enqueue_group_message(
                group_id=group_id,
                user_id=user_id,
                nickname=str(getattr(getattr(msg, "sender", None), "nickname", "")),
                text=text,
                mentions_bot=mentions_bot,
            )

    async def _enqueue_group_message(self, *, group_id: int, user_id: int, nickname: str, text: str, mentions_bot: bool) -> None:
        if not hasattr(self, "_pending_group_messages"):
            self._pending_group_messages = {}
        pending = self._pending_group_messages.get(group_id)
        if pending is None:
            pending = PendingGroupMessage(group_id=group_id)
            self._pending_group_messages[group_id] = pending
        pending.messages.append(GroupBufferedMessage(user_id=user_id, nickname=nickname, text=text, mentions_bot=mentions_bot))
        pending.mentions_bot = pending.mentions_bot or mentions_bot
        if pending.task is not None:
            pending.task.cancel()
        delay = self.runtime.config.group_chat.debounce_seconds
        logger.info(
            "group message debounced: group=%s user=%s delay=%.1fs buffered=%s mentions_bot=%s",
            group_id,
            user_id,
            delay,
            len(pending.messages),
            pending.mentions_bot,
        )
        pending.task = asyncio.create_task(self._flush_group_message_after_delay(group_id, delay))

    async def _flush_group_message_after_delay(self, group_id: int, delay: float) -> None:
        try:
            if delay > 0:
                await asyncio.sleep(delay)
            await self._flush_group_message(group_id)
        except asyncio.CancelledError:
            raise

    async def _flush_group_message(self, group_id: int) -> None:
        pending = self._pending_group_messages.pop(group_id, None)
        if pending is None:
            return
        messages = tuple(pending.messages)
        last = messages[-1] if messages else GroupBufferedMessage(0, "", "", False)
        logger.info(
            "group message debounce flush: group=%s user=%s buffered=%s latest=%r",
            group_id,
            last.user_id,
            len(messages),
            self._clip_log(last.text),
        )
        group_result = await self.runtime.handle_group_message(
            GroupMessageInput(
                group_id=pending.group_id,
                user_id=last.user_id,
                nickname=last.nickname,
                text=last.text,
                mentions_bot=pending.mentions_bot,
                buffered_messages=messages,
            )
        )
        logger.info(
            "group message handled: group=%s user=%s result=%s replied=%s reason=%s",
            pending.group_id,
            last.user_id,
            bool(group_result),
            bool(group_result and group_result.reply),
            getattr(group_result, "reason", None),
        )
        if group_result and group_result.admin_notification:
            await self._send_admin_notifications(group_result.admin_notification)
        if group_result and group_result.analysis_job_id and group_result.feedback_id:
            await self.runtime.storage.create_analysis_delivery(
                job_id=group_result.analysis_job_id,
                feedback_id=group_result.feedback_id,
                group_id=pending.group_id,
                user_id=last.user_id,
            )
            asyncio.create_task(
                self._watch_feedback_analysis_job(
                    group_result.analysis_job_id,
                    pending.group_id,
                    last.user_id,
                )
            )
        if group_result and group_result.reply:
            await self._send_group_reply(pending.group_id, group_result.reply, group_result.mention_admin_id)

    @staticmethod
    def _message_text(msg) -> str:
        return str(getattr(msg, "raw_message", "") or getattr(msg, "text", "") or DoraOpsPlugin._extract_text(msg))

    @staticmethod
    def _clip_log(text: str, limit: int = 160) -> str:
        compact = " ".join(str(text).split())
        return compact if len(compact) <= limit else f"{compact[:limit].rstrip()}..."

    @staticmethod
    def _extract_text(msg) -> str:
        message = getattr(msg, "message", []) or []
        text = getattr(message, "text", None)
        if isinstance(text, str):
            return text
        chunks: list[str] = []
        for item in message:
            if DoraOpsPlugin._segment_type(item) == "text":
                chunks.append(DoraOpsPlugin._segment_text(item))
        return "".join(chunks)

    @staticmethod
    def _mentions_bot(msg) -> bool:
        message = getattr(msg, "message", []) or []
        try:
            if message.filter_at():
                return True
        except AttributeError:
            pass
        for item in message:
            if DoraOpsPlugin._segment_type(item) == "at":
                return True
        return False

    @staticmethod
    def _segment_type(item) -> str:
        if isinstance(item, dict):
            return str(item.get("type") or "")
        to_dict = getattr(item, "to_dict", None)
        if callable(to_dict):
            data = to_dict()
            if isinstance(data, dict):
                return str(data.get("type") or "")
        value = getattr(item, "type", None) or getattr(item, "message_type", None)
        if value is not None:
            return str(value)
        name = type(item).__name__.lower()
        if "plain" in name or "text" in name:
            return "text"
        if name in {"at", "mention"} or "at" == name:
            return "at"
        return ""

    @staticmethod
    def _segment_text(item) -> str:
        if isinstance(item, dict):
            data = item.get("data", {})
            if isinstance(data, dict):
                return str(data.get("text") or "")
            return ""
        to_dict = getattr(item, "to_dict", None)
        if callable(to_dict):
            data = to_dict()
            if isinstance(data, dict):
                segment_data = data.get("data", {})
                if isinstance(segment_data, dict):
                    return str(segment_data.get("text") or "")
        for attr in ("text", "content", "data"):
            value = getattr(item, attr, None)
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                return str(value.get("text") or "")
        return ""

    async def _send_private_reply(self, user_id: int, text: str) -> None:
        await self.api.qq.post_private_msg(user_id, text=text)

    async def _send_group_reply(self, group_id: int, text: str, at_user_id: int | None = None) -> None:
        if at_user_id is not None:
            await self.api.qq.post_group_msg(group_id, text=self._text_after_at(text), at=at_user_id)
            return
        await self.api.qq.post_group_msg(group_id, text=text)

    @staticmethod
    def _text_after_at(text: str) -> str:
        return text if text.startswith((" ", "\n")) else f" {text}"

    async def _send_admin_notifications(self, text: str) -> None:
        for user_id in sorted(self.runtime.config.admin.user_ids):
            await self._send_private_reply(user_id, text)

    def _daily_summary_group_ids(self) -> list[int]:
        configured = self.runtime.config.scheduler.daily_summary_group_ids
        if configured:
            return sorted(configured)
        return sorted(self.runtime.config.group_chat.enabled_group_ids)

    async def _send_daily_progress_result_to_groups(self, text: str) -> None:
        for group_id in self._daily_summary_group_ids():
            await self._send_group_reply(group_id, text)
            await self._record_group_assistant_message(group_id, text)

    async def _record_group_assistant_message(self, group_id: int, text: str) -> None:
        await self.runtime.storage.append_chat_message(f"group:{group_id}", "assistant", text)

    async def _watch_feedback_analysis_job(self, job_id: int, group_id: int, user_id: int) -> None:
        deadline = asyncio.get_running_loop().time() + self.runtime.config.jobs.max_runtime_seconds + 30
        terminal = {JobStatus.SUCCEEDED.value, JobStatus.FAILED.value, JobStatus.TIMEOUT.value}
        while True:
            job = await self.runtime.storage.get_job(job_id)
            if job is not None:
                await self.runtime.jobs.reconcile_job(job)
                job = await self.runtime.storage.get_job(job_id)
            status = str(job.get("status") if job else "")
            if job is not None and status in terminal:
                text = await self._format_feedback_analysis_result(job)
                await self._send_group_reply(group_id, text, at_user_id=user_id or None)
                await self._record_group_assistant_message(group_id, text)
                await self.runtime.storage.mark_analysis_delivered(job_id)
                return
            if asyncio.get_running_loop().time() >= deadline:
                await self.runtime.storage.update_job_status(job_id, JobStatus.TIMEOUT, "feedback analysis watcher timed out")
                job = await self.runtime.storage.get_job(job_id)
                text = await self._format_feedback_analysis_result(job or {"id": job_id, "status": JobStatus.TIMEOUT.value, "error": "feedback analysis watcher timed out"})
                await self._send_group_reply(group_id, text, at_user_id=user_id or None)
                await self._record_group_assistant_message(group_id, text)
                await self.runtime.storage.mark_analysis_delivered(job_id)
                return
            await asyncio.sleep(5)

    async def _format_feedback_analysis_result(self, job: dict[str, object]) -> str:
        status = str(job.get("status") or "-")
        payload: dict[str, object] = {
            "job_id": int(job.get("id") or 0),
            "status": status,
        }
        if status == JobStatus.SUCCEEDED.value:
            payload["opencode_output"] = self._read_text(Path(str(job["output_path"])), limit=12000)
        elif status in {JobStatus.FAILED.value, JobStatus.TIMEOUT.value}:
            payload["error"] = str(job.get("error") or status)
        client = self._progress_summary_chat_client()
        if client is not None:
            try:
                reply = await client.complete(
                    [
                        {
                            "role": "system",
                            "content": (
                                f"{DORA_PERSONA_PROMPT}\n\n"
                                "# 反馈分析结果总结任务\n"
                                "- 你正在把 opencode 对单条群聊反馈的分析结果整理成回复提问人的 QQ 消息。\n"
                                "- 输入里的 opencode_output 是原始文本，可能是 JSON、Markdown 或普通文本，不要依赖固定 schema。\n"
                                "- 只输出中文总结，不输出 JSON、Markdown 代码块或原始字段名。\n"
                                "- 说明判断、可能原因、需要补充的信息和下一步建议；失败或超时要明确说明。\n"
                                "- 控制在 5 条以内，语气保持多萝人格，但不要影响信息准确性。"
                            ),
                        },
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
                    ]
                )
            except LLMError:
                reply = ""
            if reply.strip():
                return reply.strip()
        return self._format_feedback_analysis_result_fallback(payload)

    @staticmethod
    def _format_feedback_analysis_result_fallback(payload: dict[str, object]) -> str:
        status = str(payload.get("status") or "-")
        if status == JobStatus.SUCCEEDED.value:
            output = str(payload.get("opencode_output") or "分析完成")
            return f"分析完成：{DoraOpsPlugin._clip(output, 900)}"
        error = str(payload.get("error") or status)
        return f"分析任务{status}：{DoraOpsPlugin._clip(error, 500)}"

    def _maybe_watch_daily_progress(
        self,
        command_text: str,
        result_text: str,
        send: Callable[[str], Awaitable[object]],
    ) -> None:
        if not command_text.startswith("/test daily-summary") or "--progress" not in command_text:
            return
        job_ids = [int(match.group(1)) for match in PROGRESS_JOB_PATTERN.finditer(result_text)]
        if not job_ids:
            return
        asyncio.create_task(self._watch_progress_jobs(job_ids, send))

    def _maybe_watch_feedback_analysis(self, result_text: str) -> None:
        match = APPROVED_FEEDBACK_JOB_PATTERN.search(result_text)
        if not match:
            return
        job_id = int(match.group(1))
        asyncio.create_task(self._watch_deliverable_feedback_analysis_job(job_id))

    async def _watch_deliverable_feedback_analysis_job(self, job_id: int) -> None:
        delivery = await self.runtime.storage.get_analysis_delivery(job_id)
        if delivery is None or delivery.get("delivered_at") is not None or delivery.get("group_id") is None:
            return
        await self._watch_feedback_analysis_job(
            job_id,
            int(delivery["group_id"]),
            int(delivery["user_id"] or 0),
        )

    async def _watch_progress_jobs(
        self,
        job_ids: list[int],
        send: Callable[[str], Awaitable[object]],
    ) -> None:
        deadline = asyncio.get_running_loop().time() + self.runtime.config.jobs.max_runtime_seconds + 30
        terminal = {JobStatus.SUCCEEDED.value, JobStatus.FAILED.value, JobStatus.TIMEOUT.value}
        while True:
            jobs = await self._progress_jobs_by_id(job_ids)
            for job in jobs.values():
                await self.runtime.jobs.reconcile_job(job)
            jobs = await self._progress_jobs_by_id(job_ids)
            if len(jobs) == len(job_ids) and all(str(jobs[job_id].get("status")) in terminal for job_id in job_ids):
                await send(await self._format_progress_results(job_ids, jobs))
                return
            if asyncio.get_running_loop().time() >= deadline:
                for job_id in job_ids:
                    job = jobs.get(job_id)
                    if job is not None and str(job.get("status")) not in terminal:
                        await self.runtime.storage.update_job_status(
                            job_id,
                            JobStatus.TIMEOUT,
                            "progress watcher timed out",
                        )
                jobs = await self._progress_jobs_by_id(job_ids)
                await send(await self._format_progress_results(job_ids, jobs))
                return
            await asyncio.sleep(5)

    async def _progress_jobs_by_id(self, job_ids: list[int]) -> dict[int, dict[str, object]]:
        jobs = await self.runtime.storage.list_recent_jobs(limit=50, include_test=True)
        wanted = set(job_ids)
        return {int(job["id"]): job for job in jobs if int(job["id"]) in wanted}

    async def _format_progress_results(self, job_ids: list[int], jobs: dict[int, dict[str, object]]) -> str:
        entries = self._progress_result_entries(job_ids, jobs)
        client = self._progress_summary_chat_client()
        if client is not None:
            try:
                reply = await client.complete(
                    [
                        {
                            "role": "system",
                            "content": (
                                f"{DORA_PERSONA_PROMPT}\n\n"
                                "# 昨日进展总结任务\n"
                                "- 你正在把多个仓库的 opencode 分析结果整理成 QQ 群里的最终日报。\n"
                                "- 输入里的 opencode_output 是原始文本，可能是 JSON、Markdown 或普通文本，不要依赖固定 schema。\n"
                                "- 只输出面向群友的中文总结，不输出 JSON、Markdown 代码块或原始字段名。\n"
                                "- 重点写用户可见变化、开发者需要关注的风险、建议动作；无提交的仓库一句话带过。\n"
                                "- 控制在 8 条以内，语气保持多萝人格，但不要影响信息准确性。\n"
                                "- 如果有失败或超时任务，明确点名。不要声称执行了没有执行的操作。"
                            ),
                        },
                        {
                            "role": "user",
                            "content": json.dumps({"results": entries}, ensure_ascii=False, indent=2),
                        },
                    ]
                )
            except LLMError:
                reply = ""
            if reply.strip():
                return reply.strip()
        return self._format_progress_results_fallback(entries)

    def _progress_result_entries(self, job_ids: list[int], jobs: dict[int, dict[str, object]]) -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []
        for job_id in job_ids:
            job = jobs.get(job_id)
            if job is None:
                entries.append({"job_id": job_id, "repo": f"job #{job_id}", "status": "missing"})
                continue
            repo = self._repo_from_progress_session(str(job.get("tmux_session") or "")) or f"job #{job_id}"
            status = str(job.get("status") or "-")
            entry: dict[str, object] = {"job_id": job_id, "repo": repo, "status": status}
            if status == JobStatus.SUCCEEDED.value:
                entry["opencode_output"] = self._read_text(Path(str(job["output_path"])), limit=12000)
            elif status in {JobStatus.FAILED.value, JobStatus.TIMEOUT.value}:
                entry["error"] = str(job.get("error") or status)
            entries.append(entry)
        return entries

    def _progress_summary_chat_client(self):
        admin = getattr(self.runtime, "admin", None)
        client = getattr(admin, "chat_client", None)
        if client is not None:
            return client
        group_chat = getattr(self.runtime, "group_chat", None)
        return getattr(group_chat, "chat_client", None)

    @staticmethod
    def _format_progress_results_fallback(entries: list[dict[str, object]]) -> str:
        lines = ["昨日进展分析结果："]
        for entry in entries:
            repo = str(entry.get("repo") or f"job #{entry.get('job_id')}")
            status = str(entry.get("status") or "-")
            if status == "missing":
                lines.append(f"- {repo}: missing")
                continue
            if status == JobStatus.SUCCEEDED.value:
                output = str(entry.get("opencode_output") or "分析完成")
                lines.append(f"- {repo}: {DoraOpsPlugin._clip(output, 600)}")
            elif status in {JobStatus.FAILED.value, JobStatus.TIMEOUT.value}:
                error = str(entry.get("error") or status)
                lines.append(f"- {repo}: {status}\n  {DoraOpsPlugin._clip(error, 500)}")
            else:
                lines.append(f"- {repo}: {status}")
        return "\n".join(lines)

    @staticmethod
    def _read_text(path: Path, *, limit: int) -> str:
        if not path.exists():
            return "分析完成，但没有输出文件"
        return path.read_text(encoding="utf-8", errors="replace").strip()[:limit]

    @staticmethod
    def _clip(text: str, limit: int) -> str:
        text = " ".join(text.split())
        return text if len(text) <= limit else f"{text[:limit].rstrip()}..."

    @staticmethod
    def _repo_from_progress_session(session: str) -> str | None:
        match = PROGRESS_SESSION_PATTERN.match(session)
        return match.group(1) if match else None
