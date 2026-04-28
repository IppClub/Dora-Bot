from __future__ import annotations

import asyncio
import json
import shlex
import time
from pathlib import Path

from .config import BotConfig, resolve_path
from .models import ChangeStatus, JobStatus
from .storage import Storage


class JobManager:
    def __init__(self, base_dir: Path, config: BotConfig, storage: Storage):
        self.base_dir = base_dir
        self.config = config
        self.storage = storage

    async def create_tmux_echo_test(self, triggered_by: str | None = None) -> int:
        job_dir = self._next_job_dir("tmux_test")
        prompt = job_dir / "prompt.md"
        output = job_dir / "output.json"
        error = job_dir / "error.log"
        exit_code = job_dir / "exit_code"
        done = job_dir / "done"
        prompt.write_text("tmux smoke test\n", encoding="utf-8")
        session = f"dora_job_{int(time.time())}_{job_dir.name}"
        job_id = await self.storage.create_job(
            kind="tmux_test",
            target_type="test",
            target_id=None,
            tmux_session=session,
            prompt_path=prompt,
            output_path=output,
            error_path=error,
            exit_code_path=exit_code,
            done_path=done,
            is_test=True,
            triggered_by=triggered_by,
            trigger_source="admin_command",
        )
        command = f"printf '%s\\n' '{{\"ok\":true,\"job_id\":{job_id}}}' > {shlex.quote(str(output))}"
        await self._start_tmux_job(job_id, session, command, error, exit_code, done)
        return job_id

    async def create_opencode_repo_analysis(
        self,
        repo_key: str,
        repo_path: Path,
        change_id: int | None,
        prompt_text: str,
        *,
        is_test: bool = False,
        triggered_by: str | None = None,
    ) -> int:
        job_dir = self._next_job_dir("opencode")
        prompt = job_dir / "prompt.md"
        output = job_dir / "output.json"
        error = job_dir / "error.log"
        exit_code = job_dir / "exit_code"
        done = job_dir / "done"
        prompt.write_text(prompt_text, encoding="utf-8")
        session = f"dora_job_{int(time.time())}_{repo_key}_{job_dir.name}"
        job_id = await self.storage.create_job(
            kind="repo_diff",
            target_type="repo_change",
            target_id=change_id,
            tmux_session=session,
            prompt_path=prompt,
            output_path=output,
            error_path=error,
            exit_code_path=exit_code,
            done_path=done,
            is_test=is_test,
            triggered_by=triggered_by,
            trigger_source="admin_command" if is_test else "repo_tracker",
        )
        opencode = self.config.jobs.opencode_command
        command = f"cd {shlex.quote(str(repo_path))} && {opencode} < {shlex.quote(str(prompt))} > {shlex.quote(str(output))}"
        await self._start_tmux_job(job_id, session, command, error, exit_code, done)
        if change_id is not None:
            await self.storage.update_repo_change_status(change_id, ChangeStatus.ANALYZING)
        return job_id

    async def create_feedback_analysis(
        self,
        repo_key: str,
        repo_path: Path,
        feedback_id: int,
        prompt_text: str,
        *,
        triggered_by: str | None = None,
    ) -> int:
        job_dir = self._next_job_dir("feedback")
        prompt = job_dir / "prompt.md"
        output = job_dir / "output.json"
        error = job_dir / "error.log"
        exit_code = job_dir / "exit_code"
        done = job_dir / "done"
        prompt.write_text(prompt_text, encoding="utf-8")
        session = f"dora_job_{int(time.time())}_{repo_key}_feedback_{feedback_id}"
        job_id = await self.storage.create_job(
            kind="feedback_analysis",
            target_type="feedback",
            target_id=feedback_id,
            tmux_session=session,
            prompt_path=prompt,
            output_path=output,
            error_path=error,
            exit_code_path=exit_code,
            done_path=done,
            triggered_by=triggered_by,
            trigger_source="admin_approval",
        )
        opencode = self.config.jobs.opencode_command
        command = f"cd {shlex.quote(str(repo_path))} && {opencode} < {shlex.quote(str(prompt))} > {shlex.quote(str(output))}"
        await self._start_tmux_job(job_id, session, command, error, exit_code, done)
        return job_id

    async def reconcile_job(self, job: dict[str, object]) -> JobStatus | None:
        job_id = int(job["id"])
        done_path = Path(str(job["done_path"]))
        if not done_path.exists():
            return None

        exit_code_path = Path(str(job["exit_code_path"]))
        exit_code = exit_code_path.read_text(encoding="utf-8").strip() if exit_code_path.exists() else "1"
        if exit_code == "0":
            await self.storage.update_job_status(job_id, JobStatus.SUCCEEDED)
            if job.get("target_type") == "repo_change" and job.get("target_id"):
                analysis = self._read_analysis(Path(str(job["output_path"])))
                summary = analysis.get("summary") or analysis.get("answer") or "分析完成"
                await self.storage.update_repo_change_status(
                    int(job["target_id"]),
                    ChangeStatus.ANALYZED,
                    summary=str(summary),
                    analysis=analysis,
                )
            return JobStatus.SUCCEEDED

        err_path = Path(str(job["error_path"]))
        error = err_path.read_text(encoding="utf-8", errors="replace")[:2000] if err_path.exists() else "job failed"
        await self.storage.update_job_status(job_id, JobStatus.FAILED, error)
        if job.get("target_type") == "repo_change" and job.get("target_id"):
            await self.storage.update_repo_change_status(int(job["target_id"]), ChangeStatus.FAILED, summary=error[:300])
        return JobStatus.FAILED

    def _next_job_dir(self, prefix: str) -> Path:
        root = resolve_path(self.base_dir, self.config.paths.job_dir)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{prefix}_{int(time.time() * 1000)}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    async def _start_tmux_job(
        self,
        job_id: int,
        session: str,
        command: str,
        error_path: Path,
        exit_code_path: Path,
        done_path: Path,
    ) -> None:
        wrapper = (
            f"({command}) 2> {shlex.quote(str(error_path))}; "
            f"printf '%s' $? > {shlex.quote(str(exit_code_path))}; "
            f"touch {shlex.quote(str(done_path))}"
        )
        proc = await asyncio.create_subprocess_exec(
            "tmux",
            "new-session",
            "-d",
            "-s",
            session,
            wrapper,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            error = stderr.decode("utf-8", errors="replace").strip()
            await self.storage.update_job_status(job_id, JobStatus.FAILED, error)
            raise RuntimeError(error)
        await self.storage.update_job_status(job_id, JobStatus.RUNNING)

    @staticmethod
    def _read_analysis(path: Path) -> dict[str, object]:
        if not path.exists():
            return {"summary": "分析完成，但没有输出文件"}
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else {"summary": text}
        except json.JSONDecodeError:
            return {"summary": text[:1000], "raw": text}
