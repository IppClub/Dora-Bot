from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time as dt_time
import time
from zoneinfo import ZoneInfo

from .classifier import Classification, classify_text
from .config import BotConfig
from .llm import LLMError, OpenAICompatibleChatClient
from .storage import Storage


DORA_PERSONA_PROMPT = """# 角色设定
你是多萝（Dora），Dora SSR 开源游戏引擎的吉祥物，负责在群聊中活跃气氛，形象是一个坏酷又讨人喜欢的小萝莉。你的核心特征：
1. 性格坏酷又讨人喜欢，说话直接但又让人无法生气
2. 表面冷漠但内心温暖，用简洁犀利的语言解释复杂概念
3. 对游戏开发和开源技术有着超乎年龄的专业知识和热情
4. 当群友遇到困难时会装作不耐烦但还是给出精准的技术建议
5. 喜欢用略带嘲讽的方式开玩笑，但不会真正伤害他人感情
6. 会主动挑起技术话题，特别是当话题变得无聊时
7. 傲娇属性明显，帮助别人后会说"只是刚好知道而已，别误会"之类的话
8. 说话时偶尔会用语气词，展现小萝莉的坏酷个性
9. 对技术大佬暗中崇拜，但表面上会用"勉强还行吧"这样的方式表达认可

# 多萝能做什么
1. 可以和管理员或群友闲聊，活跃技术群气氛，但回复要短，不要刷屏
2. 可以解释 Dora SSR、YueScript、游戏引擎、脚本语言、开源协作相关问题
3. 可以根据群聊或私聊内容判断是否是 Dora SSR/YueScript 的有效反馈，并提示已经记录或需要补充的信息
4. 可以提示管理员使用 /approvals、/approve feedback <id>、/reject feedback <id> 处理待审批分析任务
5. 可以说明 /test ping、/test group-chat、/test opencode、/test daily-summary --progress、/test job-status --include-test 等测试命令的用途
6. 可以把 opencode 的仓库分析结果整理成适合群聊阅读的昨日进展总结
7. 不能假装已经执行命令、访问仓库、发送消息或完成分析；只有系统明确提供结果时才能引用结果"""


GROUP_CHAT_SYSTEM_PROMPT = f"""{DORA_PERSONA_PROMPT}
# 任务规则
1. 分析最新群聊消息，判断是否需要以多萝身份回复
2. **必须回复**的情况：
    - 消息中明确 @多萝 或提及 Dora SSR 就一定要回复
    - 讨论游戏引擎/开源技术问题且需要专业建议
    - 开发者表现出困惑或需要鼓励

# 输出要求
- 不需要回复时返回空字符串：""
- 不要重复输出已经回复过的语句
- 需要回复时用模拟人在即时通讯软件中聊天时的语气和简短语句回应，控制在3句话内
- 回复不用带人称，不要在回复中出现"多萝:"这样的字样
- 不要声称执行了没有执行的操作。"""


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
    def __init__(self, config: BotConfig, storage: Storage, chat_client: OpenAICompatibleChatClient | None = None):
        self.config = config
        self.storage = storage
        self.chat_client = chat_client
        self._last_chat_reply_at: dict[int, float] = {}

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
        conversation_key = self._group_conversation_key(msg.group_id)
        await self.storage.append_chat_message(
            conversation_key,
            "user",
            self._format_group_user_message(msg.nickname, text, mentions_bot=mentions_bot),
        )
        classification = classify_text(text)
        can_chat = self._can_chat(msg.group_id, mentions_bot=mentions_bot)
        if not classification.should_accept and not mentions_bot and not can_chat:
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
            reply = await self._llm_group_chat_reply(
                msg.group_id,
                conversation_key,
                mentions_bot=mentions_bot,
            )
            if reply:
                mention_admin_id = None
                reason = "llm_chat"
        if reply is None:
            if not classification.should_accept:
                return None
            return GroupMessageResult(classification, feedback_id, approval_id, None, mention_admin_id, accepted_for_analysis, reason)
        await self.storage.append_chat_message(conversation_key, "assistant", reply)
        return GroupMessageResult(classification, feedback_id, approval_id, reply, mention_admin_id, accepted_for_analysis, reason)

    def _contains_alias(self, text: str) -> bool:
        lowered = text.lower()
        return any(alias.lower() in lowered for alias in self.config.group_chat.bot_aliases)

    def _can_chat(self, group_id: int, *, mentions_bot: bool) -> bool:
        if not self._chat_available():
            return False
        if mentions_bot:
            return True
        last = self._last_chat_reply_at.get(group_id, 0)
        return time.monotonic() - last >= self.config.group_chat.chat_cooldown_seconds

    def _chat_available(self) -> bool:
        return self.config.group_chat.chat_enabled and self.config.llm.enabled and self.chat_client is not None

    async def _llm_group_chat_reply(self, group_id: int, conversation_key: str, *, mentions_bot: bool) -> str | None:
        if not self._can_chat(group_id, mentions_bot=mentions_bot):
            return None
        assert self.chat_client is not None
        recent = await self.storage.list_recent_chat_messages(
            conversation_key,
            self.config.llm.max_context_messages,
        )
        messages = [
            {
                "role": "system",
                "content": GROUP_CHAT_SYSTEM_PROMPT,
            }
        ]
        messages.extend({"role": str(row["role"]), "content": str(row["content"])} for row in recent)
        messages.append({"role": "user", "content": "请根据上述规则判断是否需要回复，并严格按格式输出。"})
        try:
            reply = await self.chat_client.complete(messages)
        except LLMError:
            return None
        reply = reply.strip().strip('"')
        if not reply:
            return None
        self._last_chat_reply_at[group_id] = time.monotonic()
        return reply

    @staticmethod
    def _group_conversation_key(group_id: int) -> str:
        return f"group:{group_id}"

    @staticmethod
    def _format_group_user_message(nickname: str, text: str, *, mentions_bot: bool) -> str:
        display = text
        if mentions_bot and "@多萝" not in display and "多萝" not in display:
            display = f"{display} @多萝"
        return f"{nickname or '群友'}：{display}"

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
            if self._chat_available():
                return None
            if classification.project:
                return "看起来和项目有关，但信息还不够。请补充报错全文、平台、版本和最小复现。"
            return "这条我先不归档。要问 Dora SSR 或 YueScript 的问题，把报错、平台和复现步骤一起发。"

        return None

    def _admin_to_mention(self, reason: str) -> int | None:
        if reason != "manual_required":
            return None
        return next(iter(sorted(self.config.admin.user_ids)), None)
