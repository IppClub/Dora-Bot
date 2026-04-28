from pathlib import Path
import asyncio

import pytest

from dora_ops.models import JobStatus
from dora_ops.runtime import DoraOpsRuntime


CONFIG = """
paths:
  data_dir: data
  mirror_dir: mirrors
  job_dir: jobs
  summary_dir: summaries
repositories:
  dora_ssr:
    name: Dora SSR
    remote: https://gitcode.com/ippclub/Dora-SSR.git
    default_branch: main
    watch_tags: true
    watch_paths: []
"""


@pytest.mark.asyncio
async def test_feedback_analysis_jobs_run_serially(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)
    events: list[tuple[str, int]] = []

    async def fake_start(job_id, session, command, error_path, exit_code_path, done_path):
        events.append(("start", job_id))
        await runtime.storage.update_job_status(job_id, JobStatus.RUNNING)
        assert events.count(("start", job_id)) == 1
        assert sum(1 for event, _ in events if event == "start") == sum(1 for event, _ in events if event == "end") + 1
        output_path = Path(str((await runtime.storage.get_job(job_id))["output_path"]))
        output_path.write_text('{"summary":"ok"}', encoding="utf-8")
        error_path.write_text("", encoding="utf-8")
        exit_code_path.write_text("0", encoding="utf-8")
        done_path.touch()
        events.append(("end", job_id))

    runtime.jobs._start_tmux_job = fake_start  # type: ignore[method-assign]

    first = await runtime.jobs.create_feedback_analysis("dora_ssr", tmp_path, 1, "prompt")
    second = await runtime.jobs.create_feedback_analysis("dora_ssr", tmp_path, 2, "prompt")

    for _ in range(20):
        first_job = await runtime.storage.get_job(first)
        second_job = await runtime.storage.get_job(second)
        if first_job and second_job and first_job["status"] == second_job["status"] == "succeeded":
            break
        await asyncio.sleep(0.01)
    assert events == [("start", first), ("end", first), ("start", second), ("end", second)]
