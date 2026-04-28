from dora_ops.prompts import yesterday_progress_prompt


def test_yesterday_progress_prompt_mentions_pull_and_range() -> None:
    prompt = yesterday_progress_prompt(repo_name="Dora SSR", branch="main", timezone="Asia/Shanghai")
    assert "git pull -f origin main" in prompt
    assert "yesterday 00:00" in prompt
    assert "today 00:00" in prompt
