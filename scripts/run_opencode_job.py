#!/usr/bin/env python3
"""Run an OpenCode job to a verifiable final result.

OpenCode can occasionally end a parent turn after delegating exploration.  A
zero exit status in that case is not a completed Bot job: the required JSON or
design document has not been produced.  This wrapper gives every job a stable
OpenCode session title, validates its artifact, and resumes that same session
until it returns a final artifact (within a small bounded retry budget).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sqlite3
import subprocess
import sys
import time
from pathlib import Path


CONTINUE_PROMPT = """Your previous turn ended without the required final artifact.
Do not launch background or sub-agents. First use any completed work already in
this session, then finish the original task yourself in this foreground turn.
Do not answer that you are waiting. Produce the required final JSON only, or
for a document task write and verify the exact requested file before finishing.
"""


def _json_object(text: str) -> bool:
    text = text.strip()
    candidates = [text]
    if "```" in text:
        for block in text.split("```")[1::2]:
            candidates.append(block.removeprefix("json").strip())
    for candidate in candidates:
        try:
            return isinstance(json.loads(candidate), dict)
        except json.JSONDecodeError:
            continue
    return False


def _database_path(command: list[str], environment: dict[str, str]) -> Path | None:
    executable = command[0]
    proc = subprocess.run([executable, "db", "path"], text=True, capture_output=True, check=False, env=environment)
    if proc.returncode == 0:
        candidate = Path(proc.stdout.strip())
        if candidate.is_file():
            return candidate
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    candidate = data_home / "opencode" / "opencode.db"
    return candidate if candidate.is_file() else None


def _session_id(command: list[str], title: str, environment: dict[str, str]) -> str | None:
    database = _database_path(command, environment)
    if database is None:
        return None
    try:
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            row = connection.execute(
                "SELECT id FROM session WHERE title = ? ORDER BY time_created DESC LIMIT 1", (title,)
            ).fetchone()
    except sqlite3.Error:
        return None
    return str(row[0]) if row else None


def _file_signature(path: Path | None) -> tuple[int, int] | None:
    if path is None or not path.is_file():
        return None
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def _run(command: list[str], *, stdin: Path | None = None, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    if stdin is None:
        return subprocess.run(command, text=True, capture_output=True, check=False, env=environment)
    with stdin.open("r", encoding="utf-8") as source:
        return subprocess.run(command, text=True, stdin=source, capture_output=True, check=False, env=environment)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opencode-command", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect", choices=("json", "file"), required=True)
    parser.add_argument("--expected-file", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--max-continuations", type=int, default=3)
    args = parser.parse_args()
    if args.expect == "file" and args.expected_file is None:
        parser.error("--expected-file is required when --expect=file")

    command = shlex.split(args.opencode_command)
    if not command or "run" not in command:
        parser.error("opencode command must include the run subcommand")
    title = f"dora-bot-job-{args.job_id}"
    attempts: list[str] = []
    initial_file_signature = _file_signature(args.expected_file)
    environment = os.environ.copy()
    environment["OPENCODE_CONFIG"] = str(args.config)

    initial = _run([*command, "--agent", args.agent, "--title", title], stdin=args.prompt, environment=environment)
    attempts.append(initial.stdout)
    if initial.stderr:
        sys.stderr.write(initial.stderr)
    if initial.returncode != 0:
        args.output.write_text(initial.stdout, encoding="utf-8")
        return initial.returncode

    session_id = _session_id(command, title, environment)
    for attempt in range(args.max_continuations + 1):
        latest = attempts[-1]
        complete = _json_object(latest) if args.expect == "json" else bool(
            args.expected_file
            and args.expected_file.is_file()
            and args.expected_file.stat().st_size > 0
            and _file_signature(args.expected_file) != initial_file_signature
        )
        if complete:
            args.output.write_text(latest, encoding="utf-8")
            return 0
        if not session_id:
            break
        time.sleep(min(5 * (attempt + 1), 15))
        resumed = _run([*command, "--session", session_id, CONTINUE_PROMPT], environment=environment)
        attempts.append(resumed.stdout)
        if resumed.stderr:
            sys.stderr.write(resumed.stderr)
        if resumed.returncode != 0:
            args.output.write_text(resumed.stdout, encoding="utf-8")
            return resumed.returncode

    args.output.write_text(attempts[-1] if attempts else "", encoding="utf-8")
    sys.stderr.write("OpenCode ended without the required final artifact after session continuation attempts.\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
