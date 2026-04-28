from __future__ import annotations

import asyncio
from pathlib import Path
import re
from typing import Awaitable, Callable

try:
    from ncatbot.plugin import NcatBotPlugin
    from ncatbot.core import registrar
except Exception:  # pragma: no cover - lets core tests run without NcatBot internals.
    NcatBotPlugin = object  # type: ignore[misc,assignment]
    registrar = None  # type: ignore[assignment]

from dora_ops.runtime import DoraOpsRuntime
from dora_ops.group_chat import GroupMessageInput
from dora_ops.jobs import JobManager
from dora_ops.models import JobStatus


PROGRESS_JOB_PATTERN = re.compile(r"job #(\d+)")
PROGRESS_SESSION_PATTERN = re.compile(r"dora_job_\d+_(.+)_progress$")


class DoraOpsPlugin(NcatBotPlugin):  # type: ignore[misc,valid-type]
    name = "dora_ops"
    version = "0.1.0"

    async def on_load(self) -> None:
        self.runtime = await DoraOpsRuntime.create(Path("dora-bot.yaml"))
        if hasattr(self, "add_scheduled_task"):
            self.add_scheduled_task(
                "daily_progress_report",
                self.runtime.config.scheduler.daily_summary_time,
                callback=self.daily_progress_report,
            )

    async def daily_progress_report(self) -> None:
        await self.runtime.summaries.create_yesterday_progress_jobs(self.runtime.jobs)

    if registrar is not None:

        @registrar.qq.on_private_message()
        async def on_private_message(self, event) -> None:
            text = getattr(event, "raw_message", "") or getattr(event, "text", "")
            user_id = int(getattr(event, "user_id", 0))
            result = await self.runtime.handle_admin_text(text, user_id=user_id)
            if result:
                await event.reply(text=result)
                self._maybe_watch_daily_progress(text, result, lambda message: event.reply(text=message))

        @registrar.qq.on_group_message()
        async def on_group_message(self, event) -> None:
            text = getattr(event, "raw_message", "") or self._extract_text(event)
            user_id = int(getattr(getattr(event, "sender", None), "user_id", 0) or getattr(event, "user_id", 0))
            group_id = int(getattr(event, "group_id", 0))
            group_result = await self.runtime.handle_group_message(
                GroupMessageInput(
                    group_id=group_id,
                    user_id=user_id,
                    nickname=str(getattr(getattr(event, "sender", None), "nickname", "")),
                    text=text,
                    mentions_bot=self._mentions_bot(event),
                )
            )
            if group_result and group_result.reply:
                await self._send_group_reply(group_id, group_result.reply, group_result.mention_admin_id)

    else:

        async def on_private_message(self, msg) -> None:
            text = getattr(msg, "raw_message", "") or getattr(msg, "text", "")
            user_id = int(getattr(msg, "user_id", 0))
            result = await self.runtime.handle_admin_text(text, user_id=user_id)
            if result:
                await self.api.post_private_msg(user_id, text=result)
                self._maybe_watch_daily_progress(
                    text,
                    result,
                    lambda message: self.api.post_private_msg(user_id, text=message),
                )

        async def on_group_message(self, msg) -> None:
            text = getattr(msg, "raw_message", "") or self._extract_text(msg)
            user_id = int(getattr(getattr(msg, "sender", None), "user_id", 0) or getattr(msg, "user_id", 0))
            group_id = int(getattr(msg, "group_id", 0))
            group_result = await self.runtime.handle_group_message(
                GroupMessageInput(
                    group_id=group_id,
                    user_id=user_id,
                    nickname=str(getattr(getattr(msg, "sender", None), "nickname", "")),
                    text=text,
                    mentions_bot=self._mentions_bot(msg),
                )
            )
            if group_result and group_result.reply:
                await self._send_group_reply(group_id, group_result.reply, group_result.mention_admin_id)

    @staticmethod
    def _extract_text(msg) -> str:
        chunks: list[str] = []
        for item in getattr(msg, "message", []) or []:
            if item.get("type") == "text":
                chunks.append(item.get("data", {}).get("text", ""))
        return "".join(chunks)

    @staticmethod
    def _mentions_bot(msg) -> bool:
        for item in getattr(msg, "message", []) or []:
            if item.get("type") == "at":
                return True
        return False

    async def _send_group_reply(self, group_id: int, text: str, at_user_id: int | None = None) -> None:
        if at_user_id is not None:
            await self.api.post_group_msg(group_id, text=text, at=at_user_id)
            return
        await self.api.post_group_msg(group_id, text=text)

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
            if all(str(jobs[job_id].get("status")) in terminal for job_id in job_ids if job_id in jobs):
                await send(self._format_progress_results(job_ids, jobs))
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
                await send(self._format_progress_results(job_ids, jobs))
                return
            await asyncio.sleep(5)

    async def _progress_jobs_by_id(self, job_ids: list[int]) -> dict[int, dict[str, object]]:
        jobs = await self.runtime.storage.list_recent_jobs(limit=50, include_test=True)
        wanted = set(job_ids)
        return {int(job["id"]): job for job in jobs if int(job["id"]) in wanted}

    @staticmethod
    def _format_progress_results(job_ids: list[int], jobs: dict[int, dict[str, object]]) -> str:
        lines = ["昨日进展分析结果："]
        for job_id in job_ids:
            job = jobs.get(job_id)
            if job is None:
                lines.append(f"- job #{job_id}: missing")
                continue
            repo = DoraOpsPlugin._repo_from_progress_session(str(job.get("tmux_session") or "")) or f"job #{job_id}"
            status = str(job.get("status") or "-")
            if status == JobStatus.SUCCEEDED.value:
                analysis = JobManager._read_analysis(Path(str(job["output_path"])))
                summary = analysis.get("summary") or analysis.get("announcement") or analysis.get("raw") or "分析完成"
                lines.append(f"- {repo}: succeeded\n  {str(summary)[:500]}")
            elif status in {JobStatus.FAILED.value, JobStatus.TIMEOUT.value}:
                error = str(job.get("error") or status)
                lines.append(f"- {repo}: {status}\n  {error[:500]}")
            else:
                lines.append(f"- {repo}: {status}")
        return "\n".join(lines)

    @staticmethod
    def _repo_from_progress_session(session: str) -> str | None:
        match = PROGRESS_SESSION_PATTERN.match(session)
        return match.group(1) if match else None
