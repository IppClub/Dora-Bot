from __future__ import annotations

from pathlib import Path

from .admin import AdminCommands
from .config import BotConfig, load_config, resolve_path
from .jobs import JobManager
from .repo_tracker import RepoTracker
from .storage import Storage
from .summary import SummaryService


class DoraOpsRuntime:
    def __init__(self, base_dir: Path, config: BotConfig, storage: Storage):
        self.base_dir = base_dir
        self.config = config
        self.storage = storage
        self.tracker = RepoTracker(base_dir, config, storage)
        self.jobs = JobManager(base_dir, config, storage)
        self.summaries = SummaryService(base_dir, config, storage)
        self.admin = AdminCommands(base_dir, config, storage, self.tracker, self.jobs, self.summaries)

    @classmethod
    async def create(cls, config_path: str | Path = "config.yaml") -> "DoraOpsRuntime":
        config_path = Path(config_path)
        base_dir = config_path.parent if config_path.parent != Path("") else Path.cwd()
        config = load_config(config_path)
        db_path = resolve_path(base_dir, config.paths.data_dir) / "dora_bot.sqlite3"
        storage = Storage(db_path)
        await storage.init()
        return cls(base_dir, config, storage)

    async def handle_admin_text(self, text: str, *, user_id: int, group_id: int | None = None) -> str | None:
        return await self.admin.handle(text, user_id=user_id, group_id=group_id)
