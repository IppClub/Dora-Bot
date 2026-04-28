from __future__ import annotations

from dataclasses import dataclass


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


@dataclass(frozen=True)
class Classification:
    should_accept: bool
    kind: str
    project: str | None
    confidence: float
    needs_repo_analysis: bool
    summary: str

    def to_dict(self) -> dict[str, object]:
        return {
            "should_accept": self.should_accept,
            "kind": self.kind,
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
        return Classification(True, "feedback", project, 0.82, True, text[:120])
    if project:
        return Classification(True, "project_question", project, 0.66, False, text[:120])
    if feedback_hits:
        return Classification(False, "possible_feedback_unrelated", None, 0.45, False, text[:120])
    return Classification(False, "chat", None, 0.25, False, text[:120])
