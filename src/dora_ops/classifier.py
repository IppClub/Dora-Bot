from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .llm import LLMError, OpenAICompatibleChatClient


DORA_STRONG_KEYWORDS = [
    "dora ssr",
    "dora-ssr",
    "dora-cli",
    "dora!",
]

DORA_CONTEXT_KEYWORDS = [
    "dora",
    "web ide",
    "actioneditor",
    "bodyeditor",
    "android webview",
]

YUE_STRONG_KEYWORDS = [
    "yuescript",
]

YUE_CONTEXT_KEYWORDS = [
    "yue",
    "moon script",
    "moonscript",
    "teal",
]

PROJECT_CONTEXT_KEYWORDS = [
    "web ide",
    "actioneditor",
    "bodyeditor",
    "android webview",
    "moon script",
    "moonscript",
    "teal",
    "pr",
    "pull request",
    "pull-request",
    "merge request",
    "mr",
    "issue",
    "commit",
    "commits",
    "branch",
    "ci",
    "workflow",
    "workflows",
    "github actions",
    "action",
    "actions",
    "agent",
    "coding agent",
    "tool call",
    "tool calls",
    "tool calling",
    "release",
    "tag",
    "仓库",
    "代码",
    "提交",
    "分支",
    "合并",
    "工作流",
    "流水线",
    "持续集成",
    "工具",
    "工具调用",
    "并发",
    "串行",
    "队列",
    "加锁",
    "锁",
    "事件循环",
    "消息",
    "调度",
    "拉取请求",
    "议题",
    "最近",
    "哪些",
    "技术",
    "分析",
    "定位",
    "源码",
    "实现",
    "架构",
    "模块",
    "函数",
    "接口",
    "文件",
    "编译",
    "构建",
    "运行",
    "启动",
    "加载",
    "编辑器",
    "脚本",
    "引擎",
    "渲染",
    "物理",
    "性能",
    "内存",
    "资源",
    "生命周期",
    "原因",
    "为什么",
    "怎么",
    "如何",
    "tstl",
    "lua",
    "tsx",
    "typescript",
    "wasm",
    "parser",
    "switch",
    "android",
    "ios",
    "macos",
    "windows",
    "webview",
]

FEEDBACK_KEYWORDS = [
    "报错",
    "错误",
    "失败",
    "崩溃",
    "不能",
    "无法",
    "问题",
    "异常",
    "不对",
    "不正常",
    "有问题",
    "没下来",
    "没释放",
    "泄漏",
    "卡死",
    "闪退",
    "希望",
    "建议",
    "bug",
    "crash",
    "error",
    "fail",
    "broken",
    "wrong",
    "leak",
]

REPO_ANALYSIS_KEYWORDS = [
    "仓库分析",
    "跑仓库",
    "看仓库",
    "查仓库",
    "检查仓库",
    "分析仓库",
    "分析一下",
    "帮忙分析",
    "定位",
    "查一下",
    "看一下",
    "在哪",
    "哪里",
    "哪个模块",
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
                    "description": "True when the message should be recorded for admin review, including concrete feedback and project questions.",
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
    project = _project_from_text(lowered)
    feedback_hits = [keyword for keyword in FEEDBACK_KEYWORDS if keyword in lowered]

    if feedback_hits and project:
        return Classification(True, "feedback", "record_feedback", project, 0.82, True, text[:120])
    if project:
        return Classification(True, "project_question", "record_feedback", project, 0.66, True, text[:120])
    if feedback_hits:
        return Classification(False, "possible_feedback_unrelated", "ignore", None, 0.45, False, text[:120])
    return Classification(False, "chat", "ignore", None, 0.25, False, text[:120])


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
                        "普通技术讨论和没有明确项目锚点的问题用 kind=chat、should_accept=false；如果需要回复可用 action=reply 或 answer_question，但不要记录。\n"
                        "只要有明确项目锚点，并且在问技术分析、源码实现、构建运行、最近变更、架构设计、性能、渲染、资源、复现、定位、原因或检查仓库，就用 kind=project_question、action=record_feedback、should_accept=true、needs_repo_analysis=true，不要直接回答。\n"
                        "明确关联到 Dora SSR/YueScript 的报错、崩溃、失败、bug、建议、希望、复现或仓库查询也必须 should_accept=true。\n"
                        "needs_repo_analysis 对明确项目技术问题默认为 true；只有问候、闲聊、机器人使用方式或完全不需要读项目上下文的问题才为 false。"
                        "action 表示群聊行为：普通聊天和不明确内容用 ignore；明确点名闲聊可用 reply；"
                        "有效反馈和项目问题用 record_feedback。"
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
        return _classification_from_mapping(result, original_text=text, context_text=context_text)
    except (AttributeError, LLMError, ValueError, TypeError):
        if fallback:
            return classify_text(text)
        raise


def _classification_from_mapping(value: dict[str, Any], *, original_text: str, context_text: str = "") -> Classification:
    kind = str(value.get("kind") or "chat")
    if kind not in {"feedback", "project_question", "possible_feedback_unrelated", "chat"}:
        kind = "chat"
    action = str(value.get("action") or "")
    if action not in {"ignore", "reply", "record_feedback", "answer_question"}:
        action = _default_action(kind, should_accept=bool(value.get("should_accept")))
    project_value = value.get("project")
    project = str(project_value) if project_value in {"Dora-SSR", "YueScript", "Dora-SSR/YueScript"} else None
    guarded_project = _project_from_text(f"{context_text}\n{original_text}".lower())
    if project is not None and guarded_project is None:
        project = None
    elif project is not None and guarded_project is not None:
        project = guarded_project
    elif guarded_project is not None and _has_repo_analysis_signal(original_text):
        project = guarded_project
        kind = "project_question"
    should_accept = bool(value.get("should_accept"))
    needs_repo_analysis = bool(value.get("needs_repo_analysis"))
    try:
        confidence = float(value.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(confidence, 1.0))
    summary = str(value.get("summary") or original_text[:120])[:120]
    if project is None and kind in {"feedback", "project_question"}:
        should_accept = False
        needs_repo_analysis = False
        if _has_feedback_signal(original_text):
            kind = "possible_feedback_unrelated"
            action = "ignore"
        else:
            kind = "chat"
            action = "reply" if action == "answer_question" else action
            if action not in {"ignore", "reply"}:
                action = "ignore"
        confidence = min(confidence, 0.55)
    elif kind == "project_question":
        should_accept = True
        action = "record_feedback"
        needs_repo_analysis = True
    elif should_accept:
        kind = "feedback"
        action = "record_feedback"
    return Classification(should_accept, kind, action, project, confidence, needs_repo_analysis, summary)


def _default_action(kind: str, *, should_accept: bool) -> str:
    if should_accept or kind == "feedback":
        return "record_feedback"
    if kind == "project_question":
        return "record_feedback"
    return "ignore"


def _project_from_text(lowered: str) -> str | None:
    dora_strong = any(keyword in lowered for keyword in DORA_STRONG_KEYWORDS)
    yue_strong = any(keyword in lowered for keyword in YUE_STRONG_KEYWORDS)
    dora_context = any(keyword in lowered for keyword in DORA_CONTEXT_KEYWORDS)
    yue_context = any(keyword in lowered for keyword in YUE_CONTEXT_KEYWORDS)
    project_context = any(keyword in lowered for keyword in PROJECT_CONTEXT_KEYWORDS)

    dora_hit = dora_strong or (dora_context and project_context)
    yue_hit = yue_strong or (yue_context and project_context)
    if dora_hit and yue_hit:
        return "Dora-SSR/YueScript"
    if dora_hit:
        return "Dora-SSR"
    if yue_hit:
        return "YueScript"
    return None


def _has_feedback_signal(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in FEEDBACK_KEYWORDS)


def _has_repo_analysis_signal(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in REPO_ANALYSIS_KEYWORDS)
