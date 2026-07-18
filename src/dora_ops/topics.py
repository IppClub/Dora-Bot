from __future__ import annotations

import asyncio
from datetime import date, datetime, time as datetime_time, timedelta
from pathlib import Path
import re
from typing import Any
from zoneinfo import ZoneInfo

from .config import BotConfig, resolve_path
from .llm import LLMError, OpenAICompatibleChatClient
from .storage import Storage


TOPIC_CODE_RE = re.compile(r"(?<![A-Za-z0-9_])ds(\d+)(?![A-Za-z0-9_])", re.IGNORECASE)


class TopicService:
    def __init__(self, config: BotConfig, storage: Storage):
        self.config = config
        self.storage = storage

    @property
    def enabled(self) -> bool:
        return self.config.topic_archive.enabled

    @staticmethod
    def codes_in_text(text: str) -> tuple[str, ...]:
        return tuple(dict.fromkeys(f"ds{match.group(1)}".lower() for match in TOPIC_CODE_RE.finditer(text)))

    @staticmethod
    def is_ticket_candidate(text: str) -> bool:
        lowered = text.lower()
        markers = ("issue", "feature", "bug", "需求", "功能", "问题", "设计", "方案", "改进", "支持")
        return len(text.strip()) >= 8 and any(marker in lowered for marker in markers)

    async def record_references(
        self,
        *,
        group_id: int,
        user_id: int,
        nickname: str,
        messages: tuple[Any, ...],
    ) -> tuple[str, ...]:
        recorded: list[str] = []
        for code in self.codes_in_text("\n".join(str(item.text) for item in messages)):
            ticket = await self.storage.get_topic_ticket(code)
            if ticket is None or str(ticket["status"]) == "closed":
                continue
            for item in messages:
                if code in self.codes_in_text(str(item.text)):
                    await self.storage.append_topic_message(
                        ticket_id=int(ticket["id"]),
                        group_id=group_id,
                        user_id=int(item.user_id),
                        nickname=str(item.nickname),
                        content=str(item.text).strip(),
                    )
            recorded.append(code)
        return tuple(recorded)

    async def create_ticket(
        self,
        *,
        group_id: int,
        user_id: int,
        nickname: str,
        project: str | None,
        kind: str,
        title: str,
        original_text: str,
        messages: tuple[Any, ...],
    ) -> dict[str, Any]:
        ticket = await self.storage.create_topic_ticket(
            group_id=group_id,
            user_id=user_id,
            project=project,
            kind=kind,
            title=title,
            original_text=original_text,
        )
        for item in messages:
            await self.storage.append_topic_message(
                ticket_id=int(ticket["id"]),
                group_id=group_id,
                user_id=int(item.user_id),
                nickname=str(item.nickname),
                content=str(item.text).strip(),
            )
        return ticket


class TopicArchive:
    def __init__(self, base_dir: Path, config: BotConfig):
        self.base_dir = base_dir
        self.config = config
        self._lock = asyncio.Lock()

    @property
    def root(self) -> Path:
        path = self.config.topic_archive.repository.local_path
        if path is None:
            raise ValueError("topic_archive.repository.local_path 未配置")
        return resolve_path(self.base_dir, path)

    async def _git(self, *args: str) -> str:
        proc = await asyncio.create_subprocess_exec(
            "git", "-C", str(self.root), *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(stderr.decode("utf-8", errors="replace").strip() or f"git {' '.join(args)} failed")
        return stdout.decode("utf-8", errors="replace").strip()

    async def verify_and_update(self, *, topic_path: Path, topic_markdown: str, design_path: Path, design_markdown: str, commit_message: str) -> str:
        async with self._lock:
            root = self.root
            if not (root / ".git").exists():
                raise ValueError(f"话题归档仓库不存在或不是 Git 仓库：{root}")
            configured_remote = self.config.topic_archive.repository.remote.strip()
            actual_remote = await self._git("remote", "get-url", "origin")
            if configured_remote and actual_remote != configured_remote:
                raise ValueError("话题归档仓库 origin 与配置不一致")
            branch = self.config.topic_archive.repository.branch
            current_branch = await self._git("branch", "--show-current")
            if current_branch != branch:
                raise ValueError(f"话题归档仓库当前分支不是 {branch}")
            if await self._git("status", "--porcelain"):
                raise ValueError("话题归档仓库工作区不干净")
            await self._git("fetch", "origin", branch)
            await self._git("pull", "--ff-only", "origin", branch)
            self._write(root / topic_path, topic_markdown)
            self._write(root / design_path, design_markdown)
            await self._git("add", "--", str(topic_path), str(design_path))
            if await self._git("diff", "--cached", "--name-only"):
                await self._git("commit", "-m", commit_message)
                await self._git("push", "origin", branch)
            return await self._git("rev-parse", "HEAD")

    @staticmethod
    def _write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)


class TopicReviewService:
    def __init__(
        self,
        base_dir: Path,
        config: BotConfig,
        storage: Storage,
        chat_client: OpenAICompatibleChatClient | None = None,
    ):
        self.config = config
        self.storage = storage
        self.chat_client = chat_client
        self.archive = TopicArchive(base_dir, config)

    async def review_yesterday(self) -> list[str]:
        zone = ZoneInfo(self.config.topic_archive.timezone)
        yesterday = datetime.now(zone).date() - timedelta(days=1)
        return await self.review_date(yesterday)

    async def review_date(self, review_date: date, *, force_code: str | None = None) -> list[str]:
        zone = ZoneInfo(self.config.topic_archive.timezone)
        start = datetime.combine(review_date, datetime_time.min, tzinfo=zone)
        end = start + timedelta(days=1)
        if force_code:
            ticket = await self.storage.get_topic_ticket(force_code)
            tickets = [ticket] if ticket is not None else []
        else:
            tickets = await self.storage.list_topics_for_review(
                start_ts=int(start.timestamp()),
                end_ts=int(end.timestamp()),
                review_date=review_date.isoformat(),
                limit=self.config.topic_archive.daily_review_24h_limit,
            )
        completed: list[str] = []
        for ticket in tickets:
            if ticket is None:
                continue
            messages = await self.storage.list_topic_messages(
                int(ticket["id"]),
                since_id=None if force_code else int(ticket.get("last_reviewed_message_id") or 0),
                start_ts=None if force_code else int(start.timestamp()),
                end_ts=None if force_code else int(end.timestamp()),
            )
            if not messages:
                continue
            await self._review_ticket(ticket, messages, review_date, zone)
            completed.append(str(ticket["code"]))
        return completed

    async def _review_ticket(self, ticket: dict[str, Any], messages: list[dict[str, Any]], review_date: date, zone: ZoneInfo) -> None:
        code = str(ticket["code"])
        design_path = self.config.topic_archive.design_dir / f"{code}.md"
        topic_path = self.config.topic_archive.docs_dir / f"{code}.md"
        design = await self._render_design(ticket, messages, design_path, zone)
        topic = self._render_topic(ticket, design_path)
        sync_id = await self.storage.create_topic_sync(
            ticket_id=int(ticket["id"]), review_date=review_date.isoformat(),
            from_message_id=int(messages[0]["id"]), to_message_id=int(messages[-1]["id"]), artifact_path=str(design_path),
        )
        try:
            commit = await self.archive.verify_and_update(
                topic_path=topic_path, topic_markdown=topic, design_path=design_path, design_markdown=design,
                commit_message=f"docs(topic): review {code} for {review_date.isoformat()}",
            )
        except Exception as exc:
            await self.storage.finish_topic_sync(sync_id, status="failed", error=str(exc)[:2000])
            raise
        await self.storage.finish_topic_sync(sync_id, status="succeeded", commit_sha=commit)
        await self.storage.mark_topic_reviewed(
            ticket_id=int(ticket["id"]), review_date=review_date.isoformat(), last_message_id=int(messages[-1]["id"]), commit_sha=commit,
        )

    def _render_topic(self, ticket: dict[str, Any], design_path: Path) -> str:
        return "\n".join([
            f"# {ticket['code']}：{ticket['title']}", "",
            f"- 状态：{ticket['status']}", f"- 项目：{ticket.get('project') or '未指定'}",
            f"- 创建于：{self._date(int(ticket['created_at']))}", f"- 来源群：{ticket['group_id']}", "",
            "## 原始诉求", "", str(ticket["original_text"]).strip(), "",
            "## 分析设计稿", "", f"- [{design_path}]({design_path.as_posix()})", "",
        ])

    async def _render_design(self, ticket: dict[str, Any], messages: list[dict[str, Any]], design_path: Path, zone: ZoneInfo) -> str:
        fallback = self._fallback_design(ticket, messages, zone)
        if self.chat_client is None:
            return fallback
        transcript = "\n".join(
            f"#{item['id']} {item['nickname'] or item['user_id']}：{item['content']}" for item in messages
        )
        try:
            body = (await self.chat_client.complete([
                {"role": "system", "content": "你是项目设计审阅器。只输出 Markdown 正文，使用问题定义与目标、方案与取舍、建议的实现拆分、风险与验证、待确认问题、本次讨论结论六个二级标题。没有证据时写入待确认问题，不能杜撰。"},
                {"role": "user", "content": f"工单：{ticket['code']} {ticket['title']}\n原始诉求：{ticket['original_text']}\n昨天新增讨论：\n{transcript}"},
            ])).strip()
            if body:
                return self._front_matter(ticket, messages, zone) + body + "\n"
        except LLMError:
            pass
        return fallback

    def _fallback_design(self, ticket: dict[str, Any], messages: list[dict[str, Any]], zone: ZoneInfo) -> str:
        conclusions = "\n".join(f"- #{item['id']} {item['nickname'] or item['user_id']}：{item['content']}" for item in messages)
        return self._front_matter(ticket, messages, zone) + "\n".join([
            "## 问题定义与目标", "", str(ticket["original_text"]).strip(), "",
            "## 方案与取舍", "", "- 待基于后续讨论和仓库分析确认。", "",
            "## 建议的实现拆分", "", "- 待确认。", "",
            "## 风险与验证", "", "- 待确认。", "",
            "## 待确认问题", "", "- 当前没有足够信息形成最终设计结论。", "",
            "## 本次讨论结论", "", conclusions, "",
        ])

    def _front_matter(self, ticket: dict[str, Any], messages: list[dict[str, Any]], zone: ZoneInfo) -> str:
        latest = datetime.fromtimestamp(int(messages[-1]["source_message_at"]), zone)
        now = datetime.now(zone)
        return "\n".join([
            f"# {ticket['code']}：{ticket['title']}", "",
            f"- 状态：{ticket['status']}", f"- 项目：{ticket.get('project') or '未指定'}",
            f"- 创建于：{self._date(int(ticket['created_at']))}", f"- 讨论更新至：{latest.date().isoformat()}",
            f"- 文稿更新于：{now.strftime('%Y-%m-%d %H:%M %z')}", f"- 本次审阅范围：消息 #{messages[0]['id']} - #{messages[-1]['id']}", "",
        ])

    @staticmethod
    def _date(timestamp: int) -> str:
        return datetime.fromtimestamp(timestamp).date().isoformat()
