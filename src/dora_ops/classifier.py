from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .llm import LLMError, OpenAICompatibleChatClient


DORA_KEYWORDS = [
    "dora",
    "dora ssr",
    "web ide",
    "dora-cli",
    "dora!",
    "actioneditor",
    "bodyeditor",
    "wasm",
    "android webview",
]

YUE_KEYWORDS = [
    "yuescript",
    "yue",
    "moonscript",
    "teal",
    "parser",
    "switch",
]

FEEDBACK_KEYWORDS = [
    "报错",
    "错误",
    "失败",
    "崩溃",
    "不能",
    "无法",
    "希望",
    "建议",
    "bug",
    "crash",
    "error",
    "fail",
    "broken",
]


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
                    "description": "True only when the message should be recorded as a concrete Dora SSR/YueScript feedback item.",
                },
                "kind": {
                    "type": "string",
                    "enum": ["feedback", "project_question", "possible_feedback_unrelated", "chat"],
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
    lowered = text.lower()
    dora_hits = [keyword for keyword in DORA_KEYWORDS if keyword in lowered]
    yue_hits = [keyword for keyword in YUE_KEYWORDS if keyword in lowered]
    feedback_hits = [keyword for keyword in FEEDBACK_KEYWORDS if keyword in lowered]

    project: str | None = None
    if dora_hits and yue_hits:
        project = "Dora-SSR/YueScript"
    elif dora_hits:
        project = "Dora-SSR"
    elif yue_hits:
        project = "YueScript"

    if feedback_hits and project:
        return Classification(True, "feedback", "record_feedback", project, 0.82, True, text[:120])
    if project:
        return Classification(False, "project_question", "answer_question", project, 0.66, False, text[:120])
    if feedback_hits:
        return Classification(False, "possible_feedback_unrelated", "ignore", None, 0.45, False, text[:120])
    return Classification(False, "chat", "ignore", None, 0.25, False, text[:120])


async def classify_text_with_llm(
    text: str,
    client: OpenAICompatibleChatClient | None,
    *,
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
                        "should_accept 表示是否应该记录为有效反馈，不表示是否需要回复。"
                        "解释类技术问题请用 kind=project_question 且 should_accept=false；"
                        "只有明确问题、报错、建议、需求或可跟踪事项才 should_accept=true。"
                        "action 表示群聊行为：普通聊天和不明确内容用 ignore；明确需要回答的问题用 answer_question；"
                        "明确点名闲聊可用 reply；有效反馈用 record_feedback。"
                    ),
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            tool=CLASSIFY_MESSAGE_TOOL,
            tool_name="classify_message",
        )
        return _classification_from_mapping(result, original_text=text)
    except (AttributeError, LLMError, ValueError, TypeError):
        if fallback:
            return classify_text(text)
        raise


def _classification_from_mapping(value: dict[str, Any], *, original_text: str) -> Classification:
    kind = str(value.get("kind") or "chat")
    if kind not in {"feedback", "project_question", "possible_feedback_unrelated", "chat"}:
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
    if should_accept:
        kind = "feedback"
        action = "record_feedback"
    return Classification(should_accept, kind, action, project, confidence, needs_repo_analysis, summary)


def _default_action(kind: str, *, should_accept: bool) -> str:
    if should_accept or kind == "feedback":
        return "record_feedback"
    if kind == "project_question":
        return "answer_question"
    return "ignore"
