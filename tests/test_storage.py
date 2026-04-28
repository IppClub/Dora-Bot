from pathlib import Path

import pytest

from dora_ops.models import ChangeStatus
from dora_ops.storage import Storage


@pytest.mark.asyncio
async def test_storage_repo_change_lifecycle(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "bot.sqlite3")
    await storage.init()
    await storage.upsert_repo_state("dora_ssr", "https://example.test/repo.git", "main", "abc", {"v1": "abc"})
    state = await storage.get_repo_state("dora_ssr")
    assert state is not None
    assert state["last_seen_head"] == "abc"

    change_id = await storage.create_repo_change(
        "dora_ssr",
        "abc",
        "def",
        1,
        ["README.md"],
        " README.md | 1 +",
        is_test=True,
    )
    await storage.update_repo_change_status(
        change_id,
        ChangeStatus.ANALYZED,
        summary="ok",
        analysis={"summary": "ok"},
    )
    counts = await storage.daily_counts(0, include_test=True)
    assert counts["repo_changes"] == 1
