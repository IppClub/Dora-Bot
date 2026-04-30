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

TIMEOUT_CONFIG = CONFIG + """
jobs:
  max_runtime_seconds: 0
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


@pytest.mark.asyncio
async def test_queued_feedback_analysis_can_resume_after_restart(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    prompt = job_dir / "prompt.md"
    output = job_dir / "output.json"
    error = job_dir / "error.log"
    exit_code = job_dir / "exit_code"
    done = job_dir / "done"
    prompt.write_text("prompt", encoding="utf-8")

    job_id = await runtime.storage.create_job(
        kind="feedback_analysis",
        target_type="feedback",
        target_id=46,
        tmux_session="dora_job_1777477435_dora_ssr_feedback_46",
        prompt_path=prompt,
        output_path=output,
        error_path=error,
        exit_code_path=exit_code,
        done_path=done,
    )
    started: list[int] = []

    async def fake_start(job_id, session, command, error_path, exit_code_path, done_path):
        started.append(job_id)
        assert "opencode" in command
        await runtime.storage.update_job_status(job_id, JobStatus.RUNNING)
        output.write_text('{"summary":"ok"}', encoding="utf-8")
        error_path.write_text("", encoding="utf-8")
        exit_code_path.write_text("0", encoding="utf-8")
        done_path.touch()

    runtime.jobs._start_tmux_job = fake_start  # type: ignore[method-assign]
    job = await runtime.storage.get_job(job_id)
    assert job is not None

    await runtime.jobs.resume_queued_feedback_analysis(job, tmp_path)

    for _ in range(20):
        resumed = await runtime.storage.get_job(job_id)
        if resumed and resumed["status"] == JobStatus.SUCCEEDED.value:
            break
        await asyncio.sleep(0.01)

    resumed = await runtime.storage.get_job(job_id)
    assert resumed is not None
    assert resumed["status"] == JobStatus.SUCCEEDED.value
    assert started == [job_id]


@pytest.mark.asyncio
async def test_stuck_feedback_analysis_times_out_and_releases_serial_queue(tmp_path: Path) -> None:
    config_path = tmp_path / "dora-bot.yaml"
    config_path.write_text(TIMEOUT_CONFIG, encoding="utf-8")
    runtime = await DoraOpsRuntime.create(config_path)
    started: list[int] = []
    killed: list[str] = []

    async def fake_start(job_id, session, command, error_path, exit_code_path, done_path):
        started.append(job_id)
        await runtime.storage.update_job_status(job_id, JobStatus.RUNNING)
        if len(started) == 2:
            output_path = Path(str((await runtime.storage.get_job(job_id))["output_path"]))
            output_path.write_text('{"summary":"ok"}', encoding="utf-8")
            error_path.write_text("", encoding="utf-8")
            exit_code_path.write_text("0", encoding="utf-8")
            done_path.touch()

    async def fake_tmux_session_exists(_session: str) -> bool:
        return True

    async def fake_kill(session: str) -> None:
        killed.append(session)

    runtime.jobs._start_tmux_job = fake_start  # type: ignore[method-assign]
    runtime.jobs._tmux_session_exists = fake_tmux_session_exists  # type: ignore[method-assign]
    runtime.jobs._kill_tmux_session = fake_kill  # type: ignore[method-assign]

    first = await runtime.jobs.create_feedback_analysis("dora_ssr", tmp_path, 1, "prompt")
    second = await runtime.jobs.create_feedback_analysis("dora_ssr", tmp_path, 2, "prompt")

    for _ in range(20):
        first_job = await runtime.storage.get_job(first)
        second_job = await runtime.storage.get_job(second)
        if (
            first_job
            and second_job
            and first_job["status"] == JobStatus.TIMEOUT.value
            and second_job["status"] == JobStatus.SUCCEEDED.value
        ):
            break
        await asyncio.sleep(0.01)

    first_job = await runtime.storage.get_job(first)
    second_job = await runtime.storage.get_job(second)
    assert first_job is not None
    assert second_job is not None
    assert first_job["status"] == JobStatus.TIMEOUT.value
    assert second_job["status"] == JobStatus.SUCCEEDED.value
    assert started == [first, second]
    assert len(killed) == 1
