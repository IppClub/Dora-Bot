from dora_ops.prompts import topic_design_prompt, yesterday_progress_prompt


def test_yesterday_progress_prompt_mentions_pull_and_range() -> None:
    prompt = yesterday_progress_prompt(repo_name="Dora SSR", branch="main", timezone="Asia/Shanghai")
    assert "git pull -f origin main" in prompt
    assert "yesterday 00:00" in prompt
    assert "today 00:00" in prompt


def test_topic_design_prompt_limits_opencode_writes_to_design_document() -> None:
    prompt = topic_design_prompt(
        code="ds1", title="测试", topic_path="/archive/topics/ds1.md",
        design_path="/archive/designs/ds1.md", review_date="2026-07-18",
    )
    assert "/archive/topics/ds1.md" in prompt
    assert "/archive/designs/ds1.md" in prompt
    assert "Do not modify any file in the source repository" in prompt
    assert "Do not run Git write commands" in prompt
