from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    data_dir: Path = Path("data")
    mirror_dir: Path = Path("mirrors")
    job_dir: Path = Path("jobs")
    summary_dir: Path = Path("summaries")


class AdminConfig(BaseModel):
    user_ids: set[int] = Field(default_factory=set)
    group_ids: set[int] = Field(default_factory=set)


class RepositoryConfig(BaseModel):
    name: str
    remote: str
    default_branch: str = "main"
    watch_tags: bool = True
    watch_paths: list[str] = Field(default_factory=list)


class JobsConfig(BaseModel):
    max_runtime_seconds: int = 600
    opencode_command: str = "opencode run"


class SchedulerConfig(BaseModel):
    daily_summary_time: str = "23:00"
    timezone: str = "Asia/Shanghai"


class BotConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    admin: AdminConfig = Field(default_factory=AdminConfig)
    repositories: dict[str, RepositoryConfig]
    jobs: JobsConfig = Field(default_factory=JobsConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)

    def ensure_dirs(self, base_dir: Path) -> None:
        for path in [
            self.paths.data_dir,
            self.paths.mirror_dir,
            self.paths.job_dir,
            self.paths.summary_dir,
        ]:
            resolve_path(base_dir, path).mkdir(parents=True, exist_ok=True)


def resolve_path(base_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else base_dir / path


def load_config(path: str | Path = "config.yaml") -> BotConfig:
    config_path = Path(path)
    if not config_path.exists():
        example_path = config_path.with_name("config.example.yaml")
        if example_path.exists():
            config_path = example_path
        else:
            raise FileNotFoundError(f"Config file not found: {path}")

    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cfg = BotConfig.model_validate(raw)
    cfg.ensure_dirs(config_path.parent)
    return cfg
