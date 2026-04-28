from __future__ import annotations

from datetime import datetime, time as dt_time
from pathlib import Path
from zoneinfo import ZoneInfo

from .config import BotConfig, resolve_path
from .storage import Storage


def start_of_today(tz_name: str) -> int:
    tz = ZoneInfo(tz_name)
    now = datetime.now(tz)
    start = datetime.combine(now.date(), dt_time.min, tz)
    return int(start.timestamp())


class SummaryService:
    def __init__(self, base_dir: Path, config: BotConfig, storage: Storage):
        self.base_dir = base_dir
        self.config = config
        self.storage = storage

    async def build_daily_summary(self, *, dry_run: bool = False, include_test: bool = False) -> str:
        since = start_of_today(self.config.scheduler.timezone)
        counts = await self.storage.daily_counts(since, include_test=include_test)
        lines = [
            "Dora 维护日报",
            "",
            f"- 有效反馈：{counts['feedback']} 条",
            f"- 仓库变更：{counts['repo_changes']} 条",
            f"- 分析任务：{counts['jobs']} 个",
        ]
        if counts["feedback"] == 0 and counts["repo_changes"] == 0 and counts["jobs"] == 0:
            lines.append("- 今日没有需要处理的维护信号")
        result = "\n".join(lines)

        if not dry_run:
            out_dir = resolve_path(self.base_dir, self.config.paths.summary_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            date = datetime.now(ZoneInfo(self.config.scheduler.timezone)).date().isoformat()
            (out_dir / f"{date}.md").write_text(result, encoding="utf-8")
        return result
