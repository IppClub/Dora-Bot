from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import aiosqlite

from .models import ChangeStatus, JobStatus


SCHEMA = """
create table if not exists repo_state(
  repo_key text primary key,
  remote_url text not null,
  default_branch text not null,
  last_seen_head text,
  tags_json text not null default '{}',
  last_checked_at integer
);

create table if not exists repo_change(
  id integer primary key autoincrement,
  repo_key text not null,
  old_head text,
  new_head text not null,
  commit_count integer not null default 0,
  changed_files_json text not null default '[]',
  diff_stat text not null default '',
  status text not null,
  summary text,
  analysis_json text,
  is_test integer not null default 0,
  triggered_by text,
  trigger_source text,
  created_at integer not null,
  analyzed_at integer
);

create table if not exists analysis_job(
  id integer primary key autoincrement,
  kind text not null,
  target_type text not null,
  target_id integer,
  status text not null,
  tmux_session text,
  prompt_path text not null,
  output_path text not null,
  error_path text not null,
  exit_code_path text not null,
  done_path text not null,
  is_test integer not null default 0,
  triggered_by text,
  trigger_source text,
  started_at integer,
  finished_at integer,
  error text
);

create table if not exists feedback(
  id integer primary key autoincrement,
  group_id integer,
  user_id integer,
  project text,
  kind text,
  title text,
  original_text text not null,
  normalized_summary text,
  status text not null default 'open',
  is_test integer not null default 0,
  created_at integer not null
);

create table if not exists scheduler_state(
  job_name text primary key,
  last_run_date text,
  last_run_at integer
);
"""


class Storage:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    async def init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(SCHEMA)
            await db.commit()

    async def upsert_repo_state(
        self,
        repo_key: str,
        remote_url: str,
        default_branch: str,
        head: str | None,
        tags: dict[str, str],
    ) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                insert into repo_state(repo_key, remote_url, default_branch, last_seen_head, tags_json, last_checked_at)
                values (?, ?, ?, ?, ?, ?)
                on conflict(repo_key) do update set
                  remote_url=excluded.remote_url,
                  default_branch=excluded.default_branch,
                  last_seen_head=excluded.last_seen_head,
                  tags_json=excluded.tags_json,
                  last_checked_at=excluded.last_checked_at
                """,
                (repo_key, remote_url, default_branch, head, json.dumps(tags), int(time.time())),
            )
            await db.commit()

    async def get_repo_state(self, repo_key: str) -> dict[str, Any] | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            row = await (await db.execute("select * from repo_state where repo_key=?", (repo_key,))).fetchone()
            return dict(row) if row else None

    async def create_repo_change(
        self,
        repo_key: str,
        old_head: str | None,
        new_head: str,
        commit_count: int,
        changed_files: list[str],
        diff_stat: str,
        *,
        is_test: bool = False,
        triggered_by: str | None = None,
        trigger_source: str = "repo_tracker",
    ) -> int:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                insert into repo_change(
                  repo_key, old_head, new_head, commit_count, changed_files_json,
                  diff_stat, status, is_test, triggered_by, trigger_source, created_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    repo_key,
                    old_head,
                    new_head,
                    commit_count,
                    json.dumps(changed_files),
                    diff_stat,
                    ChangeStatus.PENDING.value,
                    int(is_test),
                    triggered_by,
                    trigger_source,
                    int(time.time()),
                ),
            )
            await db.commit()
            return int(cursor.lastrowid)

    async def update_repo_change_status(
        self,
        change_id: int,
        status: ChangeStatus,
        *,
        summary: str | None = None,
        analysis: dict[str, Any] | None = None,
    ) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                update repo_change set
                  status=?,
                  summary=coalesce(?, summary),
                  analysis_json=coalesce(?, analysis_json),
                  analyzed_at=case when ? in ('analyzed', 'failed', 'timeout') then ? else analyzed_at end
                where id=?
                """,
                (
                    status.value,
                    summary,
                    json.dumps(analysis) if analysis is not None else None,
                    status.value,
                    int(time.time()),
                    change_id,
                ),
            )
            await db.commit()

    async def create_job(
        self,
        *,
        kind: str,
        target_type: str,
        target_id: int | None,
        tmux_session: str,
        prompt_path: Path,
        output_path: Path,
        error_path: Path,
        exit_code_path: Path,
        done_path: Path,
        is_test: bool = False,
        triggered_by: str | None = None,
        trigger_source: str = "worker",
    ) -> int:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                insert into analysis_job(
                  kind, target_type, target_id, status, tmux_session, prompt_path,
                  output_path, error_path, exit_code_path, done_path,
                  is_test, triggered_by, trigger_source
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    kind,
                    target_type,
                    target_id,
                    JobStatus.QUEUED.value,
                    tmux_session,
                    str(prompt_path),
                    str(output_path),
                    str(error_path),
                    str(exit_code_path),
                    str(done_path),
                    int(is_test),
                    triggered_by,
                    trigger_source,
                ),
            )
            await db.commit()
            return int(cursor.lastrowid)

    async def update_job_status(self, job_id: int, status: JobStatus, error: str | None = None) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            now = int(time.time())
            await db.execute(
                """
                update analysis_job set
                  status=?,
                  started_at=case when ?='running' then coalesce(started_at, ?) else started_at end,
                  finished_at=case when ? in ('succeeded', 'failed', 'timeout') then ? else finished_at end,
                  error=coalesce(?, error)
                where id=?
                """,
                (status.value, status.value, now, status.value, now, error, job_id),
            )
            await db.commit()

    async def list_recent_jobs(self, limit: int = 10, include_test: bool = False) -> list[dict[str, Any]]:
        where = "" if include_test else "where is_test = 0"
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            rows = await (
                await db.execute(
                    f"select * from analysis_job {where} order by id desc limit ?",
                    (limit,),
                )
            ).fetchall()
            return [dict(row) for row in rows]

    async def create_feedback(
        self,
        *,
        original_text: str,
        group_id: int | None = None,
        user_id: int | None = None,
        project: str | None = None,
        kind: str | None = None,
        title: str | None = None,
        normalized_summary: str | None = None,
        is_test: bool = False,
    ) -> int:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                insert into feedback(group_id, user_id, project, kind, title, original_text, normalized_summary, is_test, created_at)
                values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    group_id,
                    user_id,
                    project,
                    kind,
                    title,
                    original_text,
                    normalized_summary,
                    int(is_test),
                    int(time.time()),
                ),
            )
            await db.commit()
            return int(cursor.lastrowid)

    async def daily_counts(self, since_ts: int, include_test: bool = False) -> dict[str, int]:
        test_filter = "" if include_test else "and is_test = 0"
        async with aiosqlite.connect(self.db_path) as db:
            feedback_row = await (
                await db.execute(
                    f"select count(*) from feedback where created_at >= ? {test_filter}",
                    (since_ts,),
                )
            ).fetchone()
            change_row = await (
                await db.execute(
                    f"select count(*) from repo_change where created_at >= ? {test_filter}",
                    (since_ts,),
                )
            ).fetchone()
            job_row = await (
                await db.execute(
                    f"select count(*) from analysis_job where coalesce(started_at, 0) >= ? {test_filter}",
                    (since_ts,),
                )
            ).fetchone()
            return {
                "feedback": int(feedback_row[0]),
                "repo_changes": int(change_row[0]),
                "jobs": int(job_row[0]),
            }
