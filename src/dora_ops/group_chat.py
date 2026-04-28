from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

from .classifier import Classification, classify_text
from .config import BotConfig
from .storage import Storage


@dataclass(frozen=True)
class GroupMessageInput:
    group_id: int
    user_id: int
    nickname: str
    text: str
    mentions_bot: bool = False


@dataclass(frozen=True)
class GroupMessageResult:
    classification: Classification
    feedback_id: int | None = None
    approval_id: int | None = None
    reply: str | None = None
    mention_admin_id: int | None = None
    accepted_for_analysis: bool = False
    reason: str = ""


class GroupMessageService:
    def __init__(self, config: BotConfig, storage: Storage):
        self.config = config
        self.storage = storage

    async def handle(self, msg: GroupMessageInput) -> GroupMessageResult | None:
        cfg = self.config.group_chat
        if not cfg.enabled:
            return None
        if cfg.enabled_group_ids and msg.group_id not in cfg.enabled_group_ids:
            return None

        text = msg.text.strip()
        if not text:
            return None
        if text.startswith("/test"):
            return None

        mentions_bot = msg.mentions_bot or self._contains_alias(text)
        classification = classify_text(text)
        if not classification.should_accept and not mentions_bot:
            return None

        feedback_id: int | None = None
        if classification.should_accept:
            feedback_id = await self.storage.create_feedback(
                original_text=text,
                group_id=msg.group_id,
                user_id=msg.user_id,
                project=classification.project,
                kind=classification.kind,
                title=classification.summary[:40],
                normalized_summary=classification.summary,
            )

        accepted_for_analysis = False
        approval_id: int | None = None
        reason = "not_needed"
        if classification.needs_repo_analysis:
            if self.config.group_chat.auto_create_analysis_jobs:
                accepted_for_analysis, reason = await self._check_and_record_quota(msg)
            else:
                reason = "manual_required"
                if feedback_id is not None:
                    existing = await self.storage.get_pending_approval("feedback", feedback_id)
                    if existing is None:
                        approval_id = await self.storage.create_approval_request(
                            target_type="feedback",
                            target_id=feedback_id,
                            requested_by=msg.user_id,
                            requested_group_id=msg.group_id,
                            command=f"/approve feedback {feedback_id}",
                        )
                    else:
                        approval_id = int(existing["id"])

        reply = self._build_reply(
            classification=classification,
            feedback_id=feedback_id,
            mentions_bot=mentions_bot,
            accepted_for_analysis=accepted_for_analysis,
            reason=reason,
        )
        mention_admin_id = self._admin_to_mention(reason)
        if reply is None:
            return GroupMessageResult(classification, feedback_id, approval_id, None, mention_admin_id, accepted_for_analysis, reason)
        return GroupMessageResult(classification, feedback_id, approval_id, reply, mention_admin_id, accepted_for_analysis, reason)

    def _contains_alias(self, text: str) -> bool:
        lowered = text.lower()
        return any(alias.lower() in lowered for alias in self.config.group_chat.bot_aliases)

    async def _check_and_record_quota(self, msg: GroupMessageInput) -> tuple[bool, str]:
        since = self._start_of_today()
        group_key = f"group:{msg.group_id}:analysis"
        user_key = f"user:{msg.user_id}:analysis"
        group_used = await self.storage.count_quota_events(group_key, since)
        if group_used >= self.config.group_chat.daily_group_analysis_limit:
            return False, "group_quota_exceeded"
        user_used = await self.storage.count_quota_events(user_key, since)
        if user_used >= self.config.group_chat.daily_user_analysis_limit:
            return False, "user_quota_exceeded"
        await self.storage.record_quota_event(group_key, group_id=msg.group_id, user_id=msg.user_id, event_type="analysis")
        await self.storage.record_quota_event(user_key, group_id=msg.group_id, user_id=msg.user_id, event_type="analysis")
        return True, "accepted"

    def _start_of_today(self) -> int:
        tz = ZoneInfo(self.config.scheduler.timezone)
        now = datetime.now(tz)
        return int(datetime.combine(now.date(), dt_time.min, tz).timestamp())

    def _build_reply(
        self,
        *,
        classification: Classification,
        feedback_id: int | None,
        mentions_bot: bool,
        accepted_for_analysis: bool,
        reason: str,
    ) -> str | None:
        if classification.should_accept and self.config.group_chat.acknowledge_feedback:
            project = classification.project or "项目"
            if classification.needs_repo_analysis:
                if accepted_for_analysis:
                    return f"收到，已记录为 #{feedback_id}。这个像是 {project} 的有效问题，深度分析先等管理员确认。"
                if reason == "manual_required":
                    return f"收到，已记录为 #{feedback_id}。这个像是 {project} 的有效问题，管理员私聊发送 /approve feedback {feedback_id} 可批准深度分析。"
                if reason == "group_quota_exceeded":
                    return f"收到，已记录为 #{feedback_id}。今天本群深度分析额度已用完。"
                if reason == "user_quota_exceeded":
                    return f"收到，已记录为 #{feedback_id}。你今天的深度分析额度已用完。"
            return f"收到，已记录为 #{feedback_id}。"

        if mentions_bot:
            if classification.project:
                return "看起来和项目有关，但信息还不够。请补充报错全文、平台、版本和最小复现。"
            return "这条我先不归档。要问 Dora SSR 或 YueScript 的问题，把报错、平台和复现步骤一起发。"

        return None

    def _admin_to_mention(self, reason: str) -> int | None:
        if reason != "manual_required":
            return None
        return next(iter(sorted(self.config.admin.user_ids)), None)
