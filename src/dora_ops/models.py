from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ChangeStatus(StrEnum):
    PENDING = "pending_analysis"
    ANALYZING = "analyzing"
    ANALYZED = "analyzed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class RepoHead:
    repo_key: str
    head: str
    tags: dict[str, str]


@dataclass(frozen=True)
class RepoChange:
    id: int
    repo_key: str
    old_head: str | None
    new_head: str
    status: ChangeStatus
    commit_count: int = 0
    diff_stat: str = ""
