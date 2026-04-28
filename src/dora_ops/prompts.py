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
