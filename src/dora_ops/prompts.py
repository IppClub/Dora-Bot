from __future__ import annotations


def repo_diff_prompt(
    *,
    repo_name: str,
    old_head: str | None,
    new_head: str,
    commit_count: int,
    changed_files: list[str],
    diff_stat: str,
) -> str:
    old = old_head or "unknown previous head"
    changed = "\n".join(f"- {path}" for path in changed_files[:200]) or "- none captured"
    return f"""You are analyzing repository changes for {repo_name}.

Run in read-only mode. Do not edit files. Do not run destructive commands.
Do all investigation in this foreground session. Do not delegate to sub-agents or finish while waiting for another task.

Analyze the change range:
- old head: {old}
- new head: {new_head}
- commit count: {commit_count}

Changed files:
{changed}

Diff stat:
```text
{diff_stat[:12000]}
```

Return JSON only, with this shape:
{{
  "summary": "short maintainer-facing summary",
  "user_visible_changes": ["..."],
  "maintainer_notes": ["..."],
  "risks": ["..."],
  "announcement": "short QQ announcement draft, or empty string",
  "should_notify_group": false,
  "confidence": "low|medium|high"
}}
"""


def feedback_analysis_prompt(
    *,
    repo_name: str,
    project: str | None,
    kind: str | None,
    title: str | None,
    original_text: str,
    analysis_task: str | None = None,
    context_summary: str | None = None,
) -> str:
    return f"""You are analyzing a community feedback item for {repo_name}.

Run in read-only mode. Do not edit files. Do not run destructive commands.
Do all investigation in this foreground session. Do not delegate to sub-agents or finish while waiting for another task.

Feedback metadata:
- project: {project or repo_name}
- kind: {kind or "unknown"}
- title: {title or ""}

Analysis task:
```text
{(analysis_task or title or original_text)[:12000]}
```

Conversation-derived context summary:
```text
{(context_summary or "No additional summarized context.")[:4000]}
```

Original user message:
```text
{original_text[:12000]}
```

Inspect the repository only as needed and return JSON only:
{{
  "summary": "short technical summary",
  "is_valid_project_issue": true,
  "likely_area": ["..."],
  "need_more_info": false,
  "questions_for_user": ["..."],
  "maintainer_notes": ["..."],
  "suggested_reply": "short QQ reply in Chinese",
  "confidence": "low|medium|high"
}}
"""


def topic_design_prompt(
    *,
    code: str,
    title: str,
    topic_path: str,
    design_path: str,
    review_date: str,
) -> str:
    return f"""You are preparing the design document for Dora topic ticket {code}: {title}.

Your current working directory is the target source repository. Analyze its real code, APIs, history and conventions as needed.

Read the complete ticket discussion from:
{topic_path}

Write the final Markdown design document directly to this exact path:
{design_path}

The document is for review date {review_date}. It must use these sections:
- 问题定义与目标
- 现有项目分析
- 方案与取舍
- 建议的实现拆分
- 风险与验证
- 待确认问题
- 本次讨论结论

Rules:
- Treat the ticket discussion as user intent, and distinguish it from facts you verified in the repository.
- Do not invent code paths, APIs, results, or conclusions.
- Do not modify any file in the source repository. Do not run Git write commands.
- Only write the specified design Markdown file; the Bot will validate, commit and push the archive repository.
- This is a file-producing task, not a chat-only analysis: use your file-write tool to create the specified design file before responding.
- Before finishing, verify that the exact design path exists and is non-empty. Do not finish with only an explanation of your analysis.
- Do all investigation in this foreground session. Do not delegate to sub-agents or finish while waiting for another task.
"""


def yesterday_progress_prompt(*, repo_name: str, branch: str, timezone: str) -> str:
    return f"""You are preparing a daily maintainer progress report for {repo_name}.

The repository has already been updated with `git pull -f origin {branch}` before this prompt runs.
Run in read-only mode. Do not edit files. Do not run destructive commands.
Do all investigation in this foreground session. Do not delegate to sub-agents or finish while waiting for another task.

Analyze yesterday's repository changes in timezone {timezone}:
- from yesterday 00:00
- until today 00:00

Use git commands such as:
- git log --since='yesterday 00:00' --until='today 00:00' --oneline
- git diff --stat <range>
- git diff --name-only <range>
- git show --stat <commit>

Return JSON only:
{{
  "summary": "short Chinese maintainer-facing summary",
  "commits": ["..."],
  "user_visible_changes": ["..."],
  "developer_notes": ["..."],
  "risks": ["..."],
  "recommended_actions": ["..."],
  "announcement": "short QQ announcement draft, or empty string",
  "should_notify_group": false,
  "confidence": "low|medium|high"
}}
"""
