from __future__ import annotations

import asyncio
import json
from pathlib import Path

from .config import BotConfig, RepositoryConfig, resolve_path
from .storage import Storage


class GitCommandError(RuntimeError):
    pass


async def run_git(args: list[str], cwd: Path | None = None) -> str:
    proc = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise GitCommandError(stderr.decode("utf-8", errors="replace").strip())
    return stdout.decode("utf-8", errors="replace")


async def ls_remote(repo: RepositoryConfig) -> tuple[str | None, dict[str, str]]:
    output = await run_git(
        [
            "ls-remote",
            repo.remote,
            "HEAD",
            f"refs/heads/{repo.default_branch}",
            "refs/tags/*",
        ]
    )
    head: str | None = None
    tags: dict[str, str] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        sha, ref = line.split("\t", 1)
        if ref == f"refs/heads/{repo.default_branch}" or ref == "HEAD":
            head = sha
        elif ref.startswith("refs/tags/") and not ref.endswith("^{}"):
            tags[ref.removeprefix("refs/tags/")] = sha
    return head, tags


class RepoTracker:
    def __init__(self, base_dir: Path, config: BotConfig, storage: Storage):
        self.base_dir = base_dir
        self.config = config
        self.storage = storage

    async def check_repo(self, repo_key: str, *, is_test: bool = False, triggered_by: str | None = None) -> dict[str, object]:
        repo = self.config.repositories[repo_key]
        head, tags = await ls_remote(repo)
        if head is None:
            raise GitCommandError(f"No head found for {repo_key}:{repo.default_branch}")

        state = await self.storage.get_repo_state(repo_key)
        old_head = state["last_seen_head"] if state else None
        old_tags = json.loads(state["tags_json"]) if state else {}
        new_tags = sorted(set(tags) - set(old_tags))

        changed = old_head is not None and old_head != head
        change_id: int | None = None
        commit_count = 0
        changed_files: list[str] = []
        diff_stat = ""

        if changed or is_test:
            mirror = await self.ensure_mirror(repo_key, repo)
            await run_git(["fetch", "--prune", "--tags", "origin"], cwd=mirror)
            if old_head:
                commit_count = await self.commit_count(mirror, old_head, head)
                changed_files = await self.changed_files(mirror, old_head, head)
                diff_stat = await self.diff_stat(mirror, old_head, head)
            change_id = await self.storage.create_repo_change(
                repo_key,
                old_head,
                head,
                commit_count,
                changed_files,
                diff_stat,
                is_test=is_test,
                triggered_by=triggered_by,
                trigger_source="admin_command" if is_test else "repo_tracker",
            )

        if not is_test:
            await self.storage.upsert_repo_state(repo_key, repo.remote, repo.default_branch, head, tags)

        return {
            "repo_key": repo_key,
            "head": head,
            "old_head": old_head,
            "changed": changed,
            "new_tags": new_tags,
            "change_id": change_id,
            "commit_count": commit_count,
            "changed_files": changed_files,
            "diff_stat": diff_stat,
        }

    async def ensure_mirror(self, repo_key: str, repo: RepositoryConfig) -> Path:
        mirror_root = resolve_path(self.base_dir, self.config.paths.mirror_dir)
        mirror = mirror_root / repo_key
        if not (mirror / ".git").exists():
            mirror.parent.mkdir(parents=True, exist_ok=True)
            await run_git(["clone", "--no-checkout", repo.remote, str(mirror)])
        return mirror

    @staticmethod
    async def commit_count(mirror: Path, old_head: str, new_head: str) -> int:
        output = await run_git(["rev-list", "--count", f"{old_head}..{new_head}"], cwd=mirror)
        return int(output.strip() or "0")

    @staticmethod
    async def changed_files(mirror: Path, old_head: str, new_head: str) -> list[str]:
        output = await run_git(["diff", "--name-only", f"{old_head}..{new_head}"], cwd=mirror)
        return [line for line in output.splitlines() if line]

    @staticmethod
    async def diff_stat(mirror: Path, old_head: str, new_head: str) -> str:
        return await run_git(["diff", "--stat", f"{old_head}..{new_head}"], cwd=mirror)
