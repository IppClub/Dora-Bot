from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    data_dir: Path = Path("data")
    job_dir: Path = Path("jobs")
    summary_dir: Path = Path("summaries")


class AdminConfig(BaseModel):
    user_ids: set[int] = Field(default_factory=set)
    group_ids: set[int] = Field(default_factory=set)


class GroupChatConfig(BaseModel):
    enabled: bool = True
    enabled_group_ids: set[int] = Field(default_factory=set)
    bot_aliases: list[str] = Field(default_factory=lambda: ["多萝", "Dora", "Dora Bot"])
    acknowledge_feedback: bool = True
    chat_enabled: bool = True
    debounce_seconds: float = Field(default=60.0, ge=0)
    chat_cooldown_seconds: int = Field(default=120, ge=0)
    auto_analysis_24h_limit: int = 10


class WelcomeConfig(BaseModel):
    enabled: bool = False
    enabled_group_ids: set[int] = Field(default_factory=set)
    message: str = "欢迎 {name} 加入 Dora 社区！"


class RepositoryConfig(BaseModel):
    name: str
    remote: str
    local_path: Path | None = None
    default_branch: str = "main"
    watch_tags: bool = True
    watch_paths: list[str] = Field(default_factory=list)


class JobsConfig(BaseModel):
    max_runtime_seconds: int = 600
    opencode_command: str = "opencode run"


class LLMProfileConfig(BaseModel):
    provider: str = "openai-compatible"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    model: str = "deepseek-chat"
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout_seconds: int = 60


class LLMConfig(BaseModel):
    enabled: bool = False
    classifier: LLMProfileConfig = Field(default_factory=LLMProfileConfig)
    summarizer: LLMProfileConfig = Field(default_factory=LLMProfileConfig)
    chat: LLMProfileConfig = Field(default_factory=LLMProfileConfig)
    max_context_messages: int = Field(default=20, ge=1)


class SchedulerConfig(BaseModel):
    daily_summary_time: str = "08:00"
    timezone: str = "Asia/Shanghai"
    daily_summary_group_ids: set[int] = Field(default_factory=set)


class BotConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    admin: AdminConfig = Field(default_factory=AdminConfig)
    group_chat: GroupChatConfig = Field(default_factory=GroupChatConfig)
    welcome: WelcomeConfig = Field(default_factory=WelcomeConfig)
    repositories: dict[str, RepositoryConfig]
    jobs: JobsConfig = Field(default_factory=JobsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)

    def ensure_dirs(self, base_dir: Path) -> None:
        for path in [
            self.paths.data_dir,
            self.paths.job_dir,
            self.paths.summary_dir,
        ]:
            resolve_path(base_dir, path).mkdir(parents=True, exist_ok=True)


def resolve_path(base_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else base_dir / path


def repository_local_path(base_dir: Path, repo_key: str, repo: RepositoryConfig) -> Path:
    if repo.local_path is None:
        raise ValueError(f"{repo_key} 缺少 repositories.{repo_key}.local_path 配置")
    path = resolve_path(base_dir, repo.local_path)
    if not path.exists():
        raise FileNotFoundError(f"{repo_key} 本地仓库不存在：{path}")
    if not (path / ".git").exists():
        raise ValueError(f"{repo_key} local_path 不是 Git 仓库：{path}")
    return path


def load_config(path: str | Path = "dora-bot.yaml") -> BotConfig:
    config_path = Path(path)
    if not config_path.exists():
        example_path = config_path.with_name("dora-bot.example.yaml")
        if example_path.exists():
            config_path = example_path
        else:
            raise FileNotFoundError(f"Config file not found: {path}")

    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cfg = BotConfig.model_validate(raw)
    cfg.ensure_dirs(config_path.parent)
    return cfg
