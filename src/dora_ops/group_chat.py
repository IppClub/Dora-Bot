from __future__ import annotations

from dataclasses import dataclass
import logging
import time

from .classifier import Classification, classify_text_with_llm
from .config import BotConfig
from .jobs import JobManager
from .llm import LLMError, OpenAICompatibleChatClient
from .prompts import feedback_analysis_prompt
from .repo_tracker import RepoTracker
from .storage import Storage


try:
    from ncatbot.utils import get_log
except Exception:  # pragma: no cover - lets core tests run without NcatBot internals.
    get_log = None  # type: ignore[assignment]


logger = get_log("DoraOps.GroupChat") if get_log is not None else logging.getLogger(__name__)

PASSIVE_REPLY_CONFIDENCE = 0.75


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
2. 可以在系统提供分析结果后解释 Dora SSR、YueScript、游戏引擎、脚本语言、开源协作相关问题
3. 可以根据群聊或私聊内容判断是否是 Dora SSR/YueScript 的有效反馈或项目问题，并交给管理员审批仓库分析
4. 可以把 opencode 的仓库分析结果整理成适合群聊阅读的昨日进展总结
5. 不能假装已经执行命令、访问仓库、发送消息或完成分析；只有系统明确提供结果时才能引用结果

# 能力触发规则
1. 当用户提到 Dora SSR、Dora、YueScript、Yue、游戏引擎、脚本语言、Web IDE、Android/iOS/macOS/Windows 构建、WASM、编辑器、性能、渲染、物理、资源加载等技术问题时，不要直接解释；应记录为项目问题并等待管理员审批仓库分析
2. 当消息包含报错、崩溃、无法、不能、失败、bug、crash、error、fail、broken、建议、希望等词，并且能关联到 Dora SSR 或 YueScript 时，应视为可能的有效反馈
3. 有效反馈信息足够时，应该说明已经记录或可由系统记录；如果涉及仓库分析，应提示需要管理员审批，但不要在群聊里展开管理员命令用法
4. 有效反馈信息不足时，不要硬编原因；应追问报错全文、平台、版本、复现步骤、相关仓库或最小示例
5. 普通闲聊、无关内容或信息不足且无法关联项目时，不要记录为反馈；可以短句闲聊，或轻微引导对方补充 Dora SSR/YueScript 相关上下文"""


GROUP_CHAT_SYSTEM_PROMPT = f"""{DORA_PERSONA_PROMPT}
# 任务规则
1. 分析最新群聊消息，判断是否需要以多萝身份回复
2. 群聊里不要抢话。没有明确需要你时保持沉默，宁愿少回也不要刷屏
3. **必须回复**的情况：
    - 消息中明确 @多萝 且不是 Dora SSR/YueScript/游戏引擎项目问题
    - 开发者表现出困惑或需要鼓励
4. 群聊中的 `@某人` 表示消息在点名或评价这个被 @ 的人；除非目标明确是多萝或上下文明确在对多萝说话，否则不要把夸奖、感谢或调侃理解成对自己的评价
5. 如果消息只是在 @ 其他群友并评价那个人，不要代替被 @ 的人回应，也不要把自己当成被夸奖的对象

# 输出要求
- 不需要回复时返回空字符串：""
- 不要重复输出已经回复过的语句
- 需要回复时用模拟人在即时通讯软件中聊天时的语气和简短语句回应，控制在3句话内
- 回复不用带人称，不要在回复中出现"多萝:"这样的字样
- 不要声称执行了没有执行的操作。"""


@dataclass(frozen=True)
class GroupMention:
    user_id: str
    display_name: str = ""


@dataclass(frozen=True)
class GroupBufferedMessage:
    user_id: int
    nickname: str
    text: str
    mentions_bot: bool = False
    mentions: tuple[GroupMention, ...] = ()


@dataclass(frozen=True)
class GroupMessageInput:
    group_id: int
    user_id: int
    nickname: str
    text: str
    mentions_bot: bool = False
    mentions: tuple[GroupMention, ...] = ()
    buffered_messages: tuple[GroupBufferedMessage, ...] = ()


@dataclass(frozen=True)
class GroupMessageResult:
    classification: Classification
    feedback_id: int | None = None
    approval_id: int | None = None
    reply: str | None = None
    admin_notification: str | None = None
    mention_admin_id: int | None = None
    accepted_for_analysis: bool = False
    analysis_job_id: int | None = None
    reason: str = ""


class GroupMessageService:
    def __init__(
        self,
        config: BotConfig,
        storage: Storage,
        tracker: RepoTracker,
        jobs: JobManager,
        chat_client: OpenAICompatibleChatClient | None = None,
        classifier_client: OpenAICompatibleChatClient | None = None,
    ):
        self.config = config
        self.storage = storage
        self.tracker = tracker
        self.jobs = jobs
        self.chat_client = chat_client
        self.classifier_client = classifier_client
        self._last_chat_reply_at: dict[int, float] = {}

    async def handle(self, msg: GroupMessageInput) -> GroupMessageResult | None:
        cfg = self.config.group_chat
        if not cfg.enabled:
            logger.info("group chat skipped: disabled group=%s user=%s", msg.group_id, msg.user_id)
            return None
        if cfg.enabled_group_ids and msg.group_id not in cfg.enabled_group_ids:
            logger.info(
                "group chat skipped: group not enabled group=%s user=%s enabled_groups=%s",
                msg.group_id,
                msg.user_id,
                sorted(cfg.enabled_group_ids),
            )
            return None

        messages = self._input_messages(msg)
        if not messages:
            if not msg.mentions_bot:
                logger.info("group chat skipped: empty text group=%s user=%s", msg.group_id, msg.user_id)
                return None
            text = "@多萝"
            messages = (GroupBufferedMessage(msg.user_id, msg.nickname, text, True, msg.mentions),)
        text = self._classification_text(messages)
        mentions_bot = msg.mentions_bot or any(item.mentions_bot or self._contains_alias(item.text) for item in messages)
        if text.startswith("/test"):
            logger.info("group chat skipped: test command ignored group=%s user=%s", msg.group_id, msg.user_id)
            return None

        conversation_key = self._group_conversation_key(msg.group_id)
        for item in messages:
            await self.storage.append_chat_message(
                conversation_key,
                "user",
                self._format_group_user_message(item, mentions_bot=item.mentions_bot),
            )
        classification = await self._classify(text)
        logger.info(
            "group chat classified: group=%s user=%s kind=%s project=%s accept=%s repo_analysis=%s confidence=%.2f mentions=%s",
            msg.group_id,
            msg.user_id,
            classification.kind,
            classification.project,
            classification.should_accept,
            classification.needs_repo_analysis,
            classification.confidence,
            mentions_bot,
        )
        can_chat = self._can_chat(msg.group_id, mentions_bot=mentions_bot)
        should_explain = self._should_explain(classification, mentions_bot=mentions_bot)
        if not classification.should_accept and not mentions_bot and not should_explain:
            logger.info(
                "group chat skipped: no trigger group=%s user=%s can_chat=%s action=%s confidence=%.2f should_explain=%s",
                msg.group_id,
                msg.user_id,
                can_chat,
                classification.action,
                classification.confidence,
                should_explain,
            )
            return None
        if not classification.should_accept and not mentions_bot and should_explain and not can_chat:
            logger.info(
                "group chat skipped: cooldown group=%s user=%s cooldown=%ss",
                msg.group_id,
                msg.user_id,
                self.config.group_chat.chat_cooldown_seconds,
            )
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
        analysis_job_id: int | None = None
        approval_id: int | None = None
        reason = "not_needed"
        if classification.needs_repo_analysis:
            if feedback_id is not None and await self._can_auto_create_analysis():
                analysis_job_id = await self._create_feedback_analysis_job(
                    feedback_id=feedback_id,
                    classification=classification,
                    text=text,
                    triggered_by=str(msg.user_id),
                )
                accepted_for_analysis = True
                reason = "auto_accepted"
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

        admin_notification = self._build_admin_notification(
            msg=msg,
            text=text,
            classification=classification,
            feedback_id=feedback_id,
            approval_id=approval_id,
            reason=reason,
        )
        reply = None if classification.should_accept else self._build_reply(
            classification=classification,
            mentions_bot=mentions_bot,
        )
        mention_admin_id = None
        if reply is None and not classification.should_accept:
            reply = await self._llm_group_chat_reply(
                msg.group_id,
                conversation_key,
                force=mentions_bot or should_explain,
            )
            if reply:
                mention_admin_id = None
                reason = "llm_chat"
        if reply is None:
            if not classification.should_accept:
                logger.info("group chat no reply: classification did not require response group=%s user=%s", msg.group_id, msg.user_id)
                return None
            logger.info(
                "group chat recorded without reply: group=%s user=%s feedback=%s approval=%s admin_notify=%s reason=%s",
                msg.group_id,
                msg.user_id,
                feedback_id,
                approval_id,
                bool(admin_notification),
                reason,
            )
            return GroupMessageResult(
                classification=classification,
                feedback_id=feedback_id,
                approval_id=approval_id,
                reply=None,
                admin_notification=admin_notification,
                mention_admin_id=mention_admin_id,
                accepted_for_analysis=accepted_for_analysis,
                analysis_job_id=analysis_job_id,
                reason=reason,
            )
        await self.storage.append_chat_message(conversation_key, "assistant", reply)
        logger.info(
            "group chat reply ready: group=%s user=%s reason=%s feedback=%s approval=%s reply_len=%s",
            msg.group_id,
            msg.user_id,
            reason,
            feedback_id,
            approval_id,
            len(reply),
        )
        return GroupMessageResult(
            classification=classification,
            feedback_id=feedback_id,
            approval_id=approval_id,
            reply=reply,
            admin_notification=admin_notification,
            mention_admin_id=mention_admin_id,
            accepted_for_analysis=accepted_for_analysis,
            analysis_job_id=analysis_job_id,
            reason=reason,
        )

    def _contains_alias(self, text: str) -> bool:
        lowered = text.lower()
        return any(alias.lower() in lowered for alias in self.config.group_chat.bot_aliases)

    @staticmethod
    def _input_messages(msg: GroupMessageInput) -> tuple[GroupBufferedMessage, ...]:
        if msg.buffered_messages:
            return tuple(item for item in msg.buffered_messages if item.text.strip())
        text = msg.text.strip()
        if not text:
            return ()
        return (GroupBufferedMessage(msg.user_id, msg.nickname, text, msg.mentions_bot, msg.mentions),)

    @staticmethod
    def _classification_text(messages: tuple[GroupBufferedMessage, ...]) -> str:
        if len(messages) == 1:
            return messages[0].text.strip()
        return "\n".join(
            f"{GroupMessageService._sender_label(item)}: {item.text.strip()}"
            for item in messages
            if item.text.strip()
        )

    def _can_chat(self, group_id: int, *, mentions_bot: bool) -> bool:
        if not self._chat_available():
            return False
        if mentions_bot:
            return True
        last = self._last_chat_reply_at.get(group_id, 0)
        return time.monotonic() - last >= self.config.group_chat.chat_cooldown_seconds

    @staticmethod
    def _should_explain(classification: Classification, *, mentions_bot: bool) -> bool:
        if mentions_bot:
            return True
        if classification.action not in {"answer_question", "reply"}:
            return False
        if classification.kind != "project_question":
            return False
        return classification.confidence >= PASSIVE_REPLY_CONFIDENCE

    def _chat_available(self) -> bool:
        return self.config.group_chat.chat_enabled and self.config.llm.enabled and self.chat_client is not None

    async def _classify(self, text: str) -> Classification:
        client = self.classifier_client if self.config.llm.enabled else None
        return await classify_text_with_llm(text, client)

    async def _llm_group_chat_reply(self, group_id: int, conversation_key: str, *, force: bool) -> str | None:
        if not force and not self._can_chat(group_id, mentions_bot=False):
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
        except LLMError as exc:
            logger.info("group chat llm reply failed: group=%s error=%s", group_id, exc)
            return None
        reply = reply.strip().strip('"')
        if not reply:
            logger.info("group chat llm reply empty: group=%s", group_id)
            return None
        self._last_chat_reply_at[group_id] = time.monotonic()
        return reply

    @staticmethod
    def _group_conversation_key(group_id: int) -> str:
        return f"group:{group_id}"

    @staticmethod
    def _format_group_user_message(item: GroupBufferedMessage, *, mentions_bot: bool) -> str:
        display = item.text
        if mentions_bot and "@多萝" not in display and "多萝" not in display:
            display = f"{display} @多萝"
        mentions = GroupMessageService._mentions_label(item.mentions)
        suffix = f" [mentions: {mentions}]" if mentions else ""
        return f"{GroupMessageService._sender_label(item)}：{display}{suffix}"

    @staticmethod
    def _sender_label(item: GroupBufferedMessage) -> str:
        name = item.nickname.strip() if item.nickname else "群友"
        return f"{name}(QQ:{item.user_id})"

    @staticmethod
    def _mentions_label(mentions: tuple[GroupMention, ...]) -> str:
        labels: list[str] = []
        for mention in mentions:
            if mention.user_id == "all":
                labels.append("@全体成员")
                continue
            display = mention.display_name.strip() or mention.user_id
            labels.append(f"@{display}(QQ: {mention.user_id})")
        return ", ".join(labels)

    async def _can_auto_create_analysis(self) -> bool:
        since = int(time.time()) - 24 * 60 * 60
        used = await self.storage.count_jobs(kind="feedback_analysis", since_ts=since)
        return used < self.config.group_chat.auto_analysis_24h_limit

    async def _create_feedback_analysis_job(
        self,
        *,
        feedback_id: int,
        classification: Classification,
        text: str,
        triggered_by: str,
    ) -> int:
        repo_key = self._repo_key_for_project(classification.project)
        repo = self.config.repositories[repo_key]
        mirror = await self.tracker.ensure_mirror(repo_key, repo)
        prompt = feedback_analysis_prompt(
            repo_name=repo.name,
            project=classification.project,
            kind=classification.kind,
            title=classification.summary[:40],
            original_text=text,
        )
        return await self.jobs.create_feedback_analysis(
            repo_key,
            mirror,
            feedback_id,
            prompt,
            triggered_by=triggered_by,
            trigger_source="group_auto_analysis",
        )

    @staticmethod
    def _repo_key_for_project(project: object) -> str:
        return {
            "Dora-SSR": "dora_ssr",
            "YueScript": "yuescript",
            "Dora-SSR/YueScript": "dora_ssr",
        }.get(str(project or "").strip(), "dora_ssr")

    def _build_reply(
        self,
        *,
        classification: Classification,
        mentions_bot: bool,
    ) -> str | None:
        if mentions_bot:
            if self._chat_available():
                return None
            if classification.project:
                return "看起来和项目有关，但信息还不够。请补充报错全文、平台、版本和最小复现。"
            return "这条我先不归档。要问 Dora SSR 或 YueScript 的问题，把报错、平台和复现步骤一起发。"

        return None

    @staticmethod
    def _build_admin_notification(
        *,
        msg: GroupMessageInput,
        text: str,
        classification: Classification,
        feedback_id: int | None,
        approval_id: int | None,
        reason: str,
    ) -> str | None:
        if not classification.should_accept or feedback_id is None:
            return None
        lines = [
            f"群聊反馈已记录：#{feedback_id}",
            f"群：{msg.group_id}",
            f"用户：{msg.nickname or msg.user_id} ({msg.user_id})",
            f"项目：{classification.project or '-'}",
            f"摘要：{classification.summary}",
            f"原因：{reason}",
            f"原文：{text}",
        ]
        if approval_id is not None:
            lines.append(f"审批：#{approval_id}")
            lines.append(f"批准分析：/approve feedback {feedback_id}")
        elif reason == "auto_accepted":
            lines.append("分析：已自动放行，等待串行任务完成")
        return "\n".join(lines)
