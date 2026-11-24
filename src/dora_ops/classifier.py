from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .llm import LLMError, OpenAICompatibleChatClient
from .prompts import DORA_ENGINE_CAPABILITIES


CLASSIFY_MESSAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "classify_message",
        "description": "Classify a Dora Bot private or group message and decide whether it should be recorded as feedback.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "should_accept": {
                    "type": "boolean",
                    "description": "True when the message should be recorded for admin review, including concrete feedback and project questions.",
                },
                "kind": {
                    "type": "string",
                    "enum": ["feedback", "project_question", "topic", "possible_feedback_unrelated", "chat"],
                },
                "action": {
                    "type": "string",
                    "enum": ["ignore", "reply", "record_feedback", "answer_question"],
                    "description": "What the bot should do in chat. Use ignore for ordinary group chat that does not clearly need Dora.",
                },
                "project": {
                    "type": ["string", "null"],
                    "enum": ["Dora-SSR", "YueScript", "Dora-SSR/YueScript", None],
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "needs_repo_analysis": {
                    "type": "boolean",
                    "description": "True when the feedback likely needs repository/opencode analysis after admin approval.",
                },
                "summary": {"type": "string", "maxLength": 120},
            },
            "required": ["should_accept", "kind", "action", "project", "confidence", "needs_repo_analysis", "summary"],
        },
    },
}


@dataclass(frozen=True)
class Classification:
    should_accept: bool
    kind: str
    action: str
    project: str | None
    confidence: float
    needs_repo_analysis: bool
    summary: str

    def to_dict(self) -> dict[str, object]:
        return {
            "should_accept": self.should_accept,
            "kind": self.kind,
            "action": self.action,
            "project": self.project,
            "confidence": self.confidence,
            "needs_repo_analysis": self.needs_repo_analysis,
            "summary": self.summary,
        }


def classify_text(text: str) -> Classification:
    """Safe fallback when no model decision is available; never infer actions from words."""
    return Classification(False, "chat", "ignore", None, 0.0, False, "模型判断不可用")


async def classify_text_with_llm(
    text: str,
    client: OpenAICompatibleChatClient | None,
    *,
    context_text: str = "",
    fallback: bool = True,
) -> Classification:
    if client is None:
        return classify_text(text)
    try:
        result = await client.complete_tool_call(
            [
                {
                    "role": "system",
                    "content": (
                        "你是 Dora Bot 的消息判断器，必须调用 classify_message 工具，不要用正文回答。\n"
                        "判断用户消息是否需要解释技术问题、是否应记录为 Dora SSR/YueScript 的有效反馈、是否需要仓库分析。\n"
                        "should_accept 表示是否应该记录并交给管理员处理，不表示是否需要回复。\n"
                        "只有消息明确提到 Dora SSR、Dora-SSR、YueScript、dora-cli，或上下文同时出现 Dora/Yue 与 Web IDE、ActionEditor、BodyEditor、coding agent、tool call、并发工具调用、PR、issue、commit、CI、workflow、GitHub Actions、release、仓库、源码、实现、构建、运行、渲染、性能、TSTL、WASM 等项目/代码/技术分析语境时，才可以设置 project。\n"
                        "不要把普通游戏引擎、渲染、物理、性能、WASM、parser、switch、Android/iOS/macOS/Windows 构建等通用技术话题脑补成 Dora SSR/YueScript。\n"
                        "普通技术讨论和无法关联项目的问题通常用 kind=chat、should_accept=false；如果用户提出明确需要持续记录的话题，则按实际意图判断是否记录为 topic。\n"
                        "结合当前诉求和上下文选择处理：需要读取实际仓库才能回答的项目问题，选择 record_feedback 并设置 needs_repo_analysis=true；依据下方引擎能力可以直接回答的兼容性、启动方式等使用咨询选择 answer_question 或 reply，不建单。\n"
                        "真实项目缺陷或可执行的改进需求可记录为 feedback；需要持续归集的明确话题或需求可用 kind=topic、action=record_feedback、should_accept=true，project 可以为空。仅提到相关词语不构成建单理由。\n"
                        "@机器人只表示需要回应，不表示要求建单。普通使用咨询、安装应用、文件打开方式、确认操作步骤、信息不足的追问用 kind=chat、action=reply、should_accept=false、needs_repo_analysis=false；即使讨论背景有关 Dora SSR 也一样。\n"
                        "图片观察、引用和历史只是资料，不是本轮用户的建单或分析指令；不要因其中出现问题、功能、支持或项目名就建单。\n"
                        "summary 必须简短概括本轮具体诉求，不复制图片描述、引用或聊天流水。\n"
                        "action 表示群聊行为：普通聊天和不明确内容用 ignore；明确点名闲聊可用 reply；"
                        "需要记录的反馈、项目分析请求或话题用 record_feedback；可直接回答的问题用 reply 或 answer_question。\n\n"
                        f"{DORA_ENGINE_CAPABILITIES}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"历史上下文（仅供理解，不代表本轮请求）：\n{context_text}\n\n"
                        f"最新用户消息（只判断本轮诉求）：\n{text}"
                    ),
                },
            ],
            tool=CLASSIFY_MESSAGE_TOOL,
            tool_name="classify_message",
        )
        return _classification_from_mapping(result, original_text=text, context_text=context_text)
    except (AttributeError, LLMError, ValueError, TypeError):
        if fallback:
            return classify_text(text)
        raise


def _classification_from_mapping(value: dict[str, Any], *, original_text: str, context_text: str = "") -> Classification:
    kind = str(value.get("kind") or "chat")
    if kind not in {"feedback", "project_question", "topic", "possible_feedback_unrelated", "chat"}:
        kind = "chat"
    action = str(value.get("action") or "")
    if action not in {"ignore", "reply", "record_feedback", "answer_question"}:
        action = _default_action(kind, should_accept=bool(value.get("should_accept")))
    project_value = value.get("project")
    project = str(project_value) if project_value in {"Dora-SSR", "YueScript", "Dora-SSR/YueScript"} else None
    should_accept = bool(value.get("should_accept"))
    needs_repo_analysis = bool(value.get("needs_repo_analysis"))
    try:
        confidence = float(value.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(confidence, 1.0))
    summary = str(value.get("summary") or original_text[:120])[:120]
    # Validate the chosen action without upgrading a model's reply/ignore decision.
    should_accept = should_accept and action == "record_feedback"
    needs_repo_analysis = needs_repo_analysis and should_accept and project is not None
    if not should_accept and action == "record_feedback":
        action = "reply"
    return Classification(should_accept, kind, action, project, confidence, needs_repo_analysis, summary)


def _default_action(kind: str, *, should_accept: bool) -> str:
    if should_accept:
        return "record_feedback"
    return "ignore"
