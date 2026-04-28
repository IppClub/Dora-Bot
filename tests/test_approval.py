from pathlib import Path

import pytest

from dora_ops.runtime import DoraOpsRuntime


CONFIG = """
paths:
  data_dir: data
  mirror_dir: mirrors
  job_dir: jobs
  summary_dir: summaries
admin:
  user_ids: [123]
  group_ids: []
group_chat:
  enabled: true
  enabled_group_ids: []
  bot_aliases: [多萝, Dora]
  acknowledge_feedback: true
  daily_group_analysis_limit: 1
  daily_user_analysis_limit: 1
  auto_create_analysis_jobs: false
repositories:
  dora_ssr:
    name: Dora SSR
    remote: https://gitcode.com/ippclub/Dora-SSR.git
    default_branch: main
    watch_tags: true
    watch_paths: []
  yuescript:
    name: YueScript
    remote: https://gitcode.com/ippclub/YueScript.git
    default_branch: main
    watch_tags: true
    watch_paths: []
"""


@pytest.mark.asyncio
async def test_approval_list_and_reject(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)
    feedback_id = await runtime.storage.create_feedback(
        original_text="Dora SSR 报错",
        project="Dora-SSR",
        kind="feedback",
    )
    approval_id = await runtime.storage.create_approval_request(
        target_type="feedback",
        target_id=feedback_id,
        requested_by=456,
        requested_group_id=789,
        command=f"/approve feedback {feedback_id}",
    )

    listed = await runtime.handle_admin_text("/approvals", user_id=123)
    assert listed is not None
    assert f"#{approval_id}" in listed

    rejected = await runtime.handle_admin_text(f"/reject feedback {feedback_id}", user_id=123)
    assert rejected == f"已拒绝：feedback {feedback_id}"
    assert await runtime.storage.get_pending_approval("feedback", feedback_id) is None
