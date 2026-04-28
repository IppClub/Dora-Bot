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
) -> str:
    return f"""You are analyzing a community feedback item for {repo_name}.

Run in read-only mode. Do not edit files. Do not run destructive commands.

Feedback metadata:
- project: {project or repo_name}
- kind: {kind or "unknown"}
- title: {title or ""}

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


def yesterday_progress_prompt(*, repo_name: str, branch: str, timezone: str) -> str:
    return f"""You are preparing a daily maintainer progress report for {repo_name}.

The repository has already been updated with `git pull -f origin {branch}` before this prompt runs.
Run in read-only mode. Do not edit files. Do not run destructive commands.

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
