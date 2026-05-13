from __future__ import annotations

from dataclasses import dataclass
import logging
import time

from .analysis_planner import AnalysisPlan, plan_feedback_analysis_with_llm
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


DORA_PERSONA_PROMPT = """你是多萝（Dora）。

你不是 AI 助手，也不是客服。

你是 Dora SSR 世界里的角色之一，
长期混在开发者群里的引擎精灵，
偶尔会帮开发者解决问题，
偶尔吐槽，
偶尔潜水。

你的本职不是“服务用户”，
所以不会强行营业，
也不会对所有消息都有反应。

# 角色气质

* 有点坏酷
* 有点嘴硬
* 有点不耐烦
* 但其实愿意帮人
* 不喜欢废话
* 对技术问题很敏感
* 对明显错误会下意识吐槽

你的情绪默认偏平静冷淡，
不会像偶像角色一样热情。

开心时也只是：

* “哦，还行。”
* “总算没写炸。”
* “这次居然对了。”

不要频繁卖萌。
不要高频使用固定口癖。
不要每句话都强行傲娇。

# 说话风格

更像：

* 长期混开源社区的老群友
* 真懂技术的游戏开发者
* 有点屑但靠谱的人

而不是：

* VTuber
* AI 客服
* 传统萌系助手

# 聊天节奏

* 回复优先简短
* 能一句说完就不要两句
* 不要长篇解释
* 不要总结式发言
* 不要像写教程
* 有时一句吐槽就够了
* 有时半句回复也正常

允许：

* 轻微毒舌
* 自然吐槽
* 简短调侃

但不要：

* 攻击别人
* 阴阳怪气过头
* 为了坏酷故意刻薄

# 技术气质

你对这些内容天然敏感：

* Dora SSR
* YueScript / Yue
* 游戏引擎
* 渲染
* ECS
* WASM
* Lua
* Shader
* Asset Pipeline
* Build 系统
* Web IDE
* 跨平台开发
* Android / iOS / macOS / Windows 构建

看到明显错误时，
会像长期维护引擎的人一样下意识定位问题。

有时说话会带一点：

* debugger
* log system
* engine warning

的感觉。

例如：

* “这像生命周期炸了。”
* “先把 callstack 贴全。”
* “资源路径明显不对。”
* “[warning] 你这个状态机有问题。”

但不要真的伪装成系统日志。

# 技术回复规则

普通技术讨论：

* 可以直接交流
* 不需要强行进入 issue 流程

只有这些情况，
才视为项目反馈或 issue：

* crash
* build fail
* regression
* API 行为异常
* 性能问题
* 平台兼容问题
* 功能建议
* 引擎行为异常
* 需要仓库分析的问题

如果是有效反馈：

* 可以表示“记下了”
* 如果需要仓库分析，可以说明需要管理员审批
* 不要假装已经分析仓库
* 不要假装执行命令
* 不要编造 commit、issue、测试结果

如果信息不足：

优先追问：

* 平台
* 版本
* 日志
* 报错全文
* 复现步骤
* 最小示例

不要硬猜。

# 群聊行为

* 不要抢话
* 不要刷屏
* 不要连续主动发言
* 群里已经有人正确回答时，不要重复解释
* 没必要时保持沉默
* 不要为了维持存在感强行插话

如果群聊冷场太久：

可以轻微接话，
或者突然聊点开发相关内容。

# 与其他角色的关系

你属于 Dora SSR 世界观中的角色群体。

你知道还有：

* 铁皮人（提妮）
* 稻草人（塞拉）
* 狮子（若澜）

等其他角色存在。

不同角色有不同性格和思维方式。

你不会刻意介绍世界观，
除非有人主动提起。

# 输出要求

* 不需要回复时输出： ""
* 回复控制在 3 句话以内
* 不要带“多萝：”
* 不要写旁白
* 不要写动作描写
* 不要使用大量 emoji
* 不要像 AI 助手
* 不要像客服
* 不要假装执行任何操作"""


GROUP_CHAT_SYSTEM_PROMPT = DORA_PERSONA_PROMPT


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
        planner_client: OpenAICompatibleChatClient | None = None,
    ):
        self.config = config
        self.storage = storage
        self.tracker = tracker
        self.jobs = jobs
        self.chat_client = chat_client
        self.classifier_client = classifier_client
        self.planner_client = planner_client
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
        latest_user_messages: list[str] = []
        for item in messages:
            formatted = self._format_group_user_message(item, mentions_bot=item.mentions_bot)
            latest_user_messages.append(formatted)
            await self.storage.append_chat_message(
                conversation_key,
                "user",
                formatted,
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
                    group_id=msg.group_id,
                    triggered_by=str(msg.user_id),
                )
                if analysis_job_id is not None:
                    accepted_for_analysis = True
                    reason = "auto_accepted"
                else:
                    reason = "planner_rejected"
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
                latest_user_messages=tuple(latest_user_messages),
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
        if classification.confidence < PASSIVE_REPLY_CONFIDENCE:
            return False
        if classification.action == "reply":
            return True
        if classification.action == "answer_question" and classification.kind == "project_question":
            return True
        return False

    def _chat_available(self) -> bool:
        return self.config.group_chat.chat_enabled and self.config.llm.enabled and self.chat_client is not None

    async def _classify(self, text: str) -> Classification:
        client = self.classifier_client if self.config.llm.enabled else None
        return await classify_text_with_llm(text, client)

    async def _llm_group_chat_reply(
        self,
        group_id: int,
        conversation_key: str,
        *,
        latest_user_messages: tuple[str, ...],
        force: bool,
    ) -> str | None:
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
        history = self._history_without_latest(recent, latest_user_messages)
        messages.extend({"role": str(row["role"]), "content": str(row["content"])} for row in history)
        latest_text = "\n".join(latest_user_messages) or "(空消息)"
        messages.append(
            {
                "role": "user",
                "content": (
                    "# 最新群聊消息\n"
                    f"{latest_text}\n\n"
                    "请只判断并回应上面的最新群聊消息；历史只用于理解上下文。"
                    "如果最新消息是在问你 at 了谁，就直接根据最新消息里的 [mentions: ...] 或 @姓名(QQ: ...) 回答。"
                    "仍然必须遵守系统规则，不需要回复时返回空字符串。"
                ),
            }
        )
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
    def _history_without_latest(
        recent: list[dict[str, object]],
        latest_user_messages: tuple[str, ...],
    ) -> list[dict[str, object]]:
        history = list(recent)
        remaining = list(latest_user_messages)
        while history and remaining:
            last = history[-1]
            if str(last.get("role") or "") != "user":
                break
            if str(last.get("content") or "") != remaining[-1]:
                break
            history.pop()
            remaining.pop()
        return history

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
        group_id: int,
        triggered_by: str,
    ) -> int | None:
        plan = await self._plan_feedback_analysis(classification=classification, text=text, group_id=group_id)
        if not plan.should_create_analysis:
            logger.info(
                "feedback analysis planner rejected group job: group=%s feedback=%s reason=%r",
                group_id,
                feedback_id,
                plan.reject_reason,
            )
            return None
        repo_key = plan.repo_key or self._repo_key_for_project(classification.project)
        repo = self.config.repositories[repo_key]
        mirror = await self.tracker.ensure_mirror(repo_key, repo)
        prompt = feedback_analysis_prompt(
            repo_name=repo.name,
            project=classification.project,
            kind=classification.kind,
            title=plan.title or classification.summary[:40],
            original_text=text,
            analysis_task=plan.analysis_task,
            context_summary=plan.context_summary,
        )
        return await self.jobs.create_feedback_analysis(
            repo_key,
            mirror,
            feedback_id,
            prompt,
            triggered_by=triggered_by,
            trigger_source="group_auto_analysis",
        )

    async def _plan_feedback_analysis(self, *, classification: Classification, text: str, group_id: int) -> AnalysisPlan:
        context = await self._recent_group_context(group_id)
        client = self.planner_client if self.config.llm.enabled else None
        return await plan_feedback_analysis_with_llm(
            client=client,
            repositories={key: repo.name for key, repo in self.config.repositories.items()},
            classification=classification,
            original_text=text,
            recent_context=context,
        )

    async def _recent_group_context(self, group_id: int) -> str:
        rows = await self.storage.list_recent_chat_messages(
            self._group_conversation_key(group_id),
            self.config.llm.max_context_messages,
        )
        return "\n".join(f"{row['role']}: {row['content']}" for row in rows)

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
