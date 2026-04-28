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
async def test_admin_ping_and_classify(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)

    denied = await runtime.handle_admin_text("/test ping", user_id=999)
    assert denied == "没有管理员权限。"

    pong = await runtime.handle_admin_text("/test ping", user_id=123)
    assert pong is not None
    assert "pong" in pong

    classified = await runtime.handle_admin_text("/test classify YueScript switch 报错", user_id=123)
    assert classified is not None
    assert "YueScript" in classified
