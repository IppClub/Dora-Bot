from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .classifier import Classification
from .llm import LLMError, OpenAICompatibleChatClient


ANALYSIS_PLAN_TOOL = {
    "type": "function",
    "function": {
        "name": "plan_feedback_analysis",
        "description": "Decide whether a feedback item should create an opencode repository analysis job and prepare the exact task.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "should_create_analysis": {
                    "type": "boolean",
                    "description": "False when the surrounding conversation shows this is not a real repository analysis request.",
                },
                "repo_key": {
                    "type": ["string", "null"],
                    "enum": ["dora_ssr", "yuescript", None],
                    "description": "Repository to inspect if analysis should be created.",
                },
                "title": {"type": "string", "maxLength": 80},
                "analysis_task": {
                    "type": "string",
                    "description": "Concrete opencode task in Chinese, derived from the conversation instead of raw chat logs.",
                },
                "context_summary": {
                    "type": "string",
                    "description": "Concise summary of the relevant chat context; do not include unrelated chat transcript.",
                },
                "reject_reason": {
                    "type": "string",
                    "description": "Reason to skip analysis when should_create_analysis is false.",
                },
                "questions_for_user": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            },
            "required": [
                "should_create_analysis",
                "repo_key",
                "title",
                "analysis_task",
                "context_summary",
                "reject_reason",
                "questions_for_user",
                "confidence",
            ],
        },
    },
}


@dataclass(frozen=True)
class AnalysisPlan:
    should_create_analysis: bool
    repo_key: str | None
    title: str
    analysis_task: str
    context_summary: str
    reject_reason: str
    questions_for_user: tuple[str, ...]
    confidence: str


async def plan_feedback_analysis_with_llm(
    *,
    client: OpenAICompatibleChatClient | None,
    repositories: dict[str, str],
    classification: Classification,
    original_text: str,
    recent_context: str,
    fallback: bool = True,
) -> AnalysisPlan:
    if client is None:
        return fallback_analysis_plan(classification=classification, original_text=original_text, repositories=repositories)
    repo_lines = "\n".join(f"- {key}: {name}" for key, name in repositories.items())
    try:
        value = await client.complete_tool_call(
            [
                {
                    "role": "system",
                    "content": (
                        "你是 Dora Bot 的仓库分析任务规划器，必须调用 plan_feedback_analysis 工具，不要用正文回答。\n"
                        "第一阶段分类器只做轻量路由，可能会误判。你需要结合最近群聊上下文二次确认：\n"
                        "1. 是否真的应该创建仓库分析任务。\n"
                        "2. 应该分析哪个仓库。\n"
                        "3. 给 opencode 的明确任务是什么。\n"
                        "不要把原始聊天流水复制进 analysis_task；只输出整理后的具体任务。\n"
                        "如果第一阶段已经给出 project 且 needs_repo_analysis=true，并且消息里有明确技术对象、源码对象、构建/运行对象、工具调用/并发对象、报错或变更查询，默认必须创建仓库分析任务。\n"
                        "只有上下文明确显示用户只是在闲聊、调侃，或本轮只是“要不要跑仓库分析”这类没有具体技术目标的泛追问时，才返回 should_create_analysis=false。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "# 可选仓库\n"
                        f"{repo_lines}\n\n"
                        "# 第一阶段分类结果\n"
                        f"- kind: {classification.kind}\n"
                        f"- project: {classification.project or '-'}\n"
                        f"- summary: {classification.summary}\n"
                        f"- needs_repo_analysis: {classification.needs_repo_analysis}\n\n"
                        "# 本次触发消息\n"
                        "```text\n"
                        f"{original_text[:8000]}\n"
                        "```\n\n"
                        "# 最近群聊上下文，按时间从旧到新\n"
                        "```text\n"
                        f"{recent_context[:12000] if recent_context else '(无)'}\n"
                        "```\n\n"
                        "请只根据与本次触发消息相关的上下文规划任务。"
                    ),
                },
            ],
            tool=ANALYSIS_PLAN_TOOL,
            tool_name="plan_feedback_analysis",
        )
        return _analysis_plan_from_mapping(value, classification=classification, original_text=original_text, repositories=repositories)
    except (AttributeError, LLMError, ValueError, TypeError):
        if fallback:
            return fallback_analysis_plan(classification=classification, original_text=original_text, repositories=repositories)
        raise


def fallback_analysis_plan(
    *,
    classification: Classification,
    original_text: str,
    repositories: dict[str, str],
) -> AnalysisPlan:
    repo_key = _repo_key_for_project(classification.project, repositories)
    title = (classification.summary or original_text).strip()[:80]
    return AnalysisPlan(
        should_create_analysis=True,
        repo_key=repo_key,
        title=title,
        analysis_task=(classification.summary or original_text).strip()[:4000],
        context_summary="未启用二次规划，使用第一阶段分类结果创建分析任务。",
        reject_reason="",
        questions_for_user=(),
        confidence="low",
    )


def _analysis_plan_from_mapping(
    value: dict[str, Any],
    *,
    classification: Classification,
    original_text: str,
    repositories: dict[str, str],
) -> AnalysisPlan:
    should_create = bool(value.get("should_create_analysis"))
    repo_key_value = value.get("repo_key")
    repo_key = str(repo_key_value) if repo_key_value in repositories else None
    if not should_create and _should_force_analysis(classification, original_text):
        should_create = True
    if should_create and repo_key is None:
        repo_key = _repo_key_for_project(classification.project, repositories)
    title = str(value.get("title") or classification.summary or original_text).strip()[:80]
    analysis_task = str(value.get("analysis_task") or classification.summary or original_text).strip()[:4000]
    context_summary = str(value.get("context_summary") or "").strip()[:2000]
    reject_reason = str(value.get("reject_reason") or "").strip()[:1000]
    questions_value = value.get("questions_for_user")
    questions = tuple(str(item).strip()[:200] for item in questions_value if str(item).strip()) if isinstance(questions_value, list) else ()
    confidence = str(value.get("confidence") or "medium")
    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"
    return AnalysisPlan(
        should_create_analysis=should_create,
        repo_key=repo_key,
        title=title,
        analysis_task=analysis_task,
        context_summary=context_summary,
        reject_reason=reject_reason,
        questions_for_user=questions,
        confidence=confidence,
    )


def _repo_key_for_project(project: object, repositories: dict[str, str]) -> str:
    mapped = {
        "Dora-SSR": "dora_ssr",
        "YueScript": "yuescript",
        "Dora-SSR/YueScript": "dora_ssr",
    }.get(str(project or "").strip(), "dora_ssr")
    return mapped if mapped in repositories else next(iter(repositories))


CONCRETE_ANALYSIS_KEYWORDS = [
    "agent",
    "coding agent",
    "tool call",
    "tool calls",
    "tool calling",
    "web ide",
    "actioneditor",
    "bodyeditor",
    "workflow",
    "github actions",
    "ci",
    "pr",
    "issue",
    "commit",
    "release",
    "tstl",
    "wasm",
    "parser",
    "switch",
    "android",
    "ios",
    "源码",
    "实现",
    "构建",
    "运行",
    "编译",
    "工具",
    "工具调用",
    "并发",
    "串行",
    "队列",
    "加锁",
    "事件循环",
    "渲染",
    "物理",
    "资源",
    "内存",
    "性能",
    "模块",
    "函数",
    "接口",
    "文件",
    "报错",
    "错误",
    "失败",
    "无法",
    "不能",
    "异常",
    "问题",
    "最近",
    "提交",
    "分支",
]


def _should_force_analysis(classification: Classification, original_text: str) -> bool:
    if not classification.project or not classification.needs_repo_analysis:
        return False
    text = f"{original_text}\n{classification.summary}".lower()
    return any(keyword in text for keyword in CONCRETE_ANALYSIS_KEYWORDS)
