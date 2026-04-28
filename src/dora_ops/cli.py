from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .runtime import DoraOpsRuntime


async def async_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    admin = sub.add_parser("admin", help="Run an admin test command")
    admin.add_argument("text", nargs="+")
    admin.add_argument("--user-id", type=int, default=0)
    admin.add_argument("--group-id", type=int)

    check = sub.add_parser("repo-check", help="Check a configured GitCode repository")
    check.add_argument("repo_key")

    args = parser.parse_args()
    runtime = await DoraOpsRuntime.create(Path(args.config))

    if args.command == "admin":
        text = " ".join(args.text)
        result = await runtime.handle_admin_text(text, user_id=args.user_id, group_id=args.group_id)
        print(result or "")
    elif args.command == "repo-check":
        result = await runtime.tracker.check_repo(args.repo_key)
        print(result)


def main() -> None:
    asyncio.run(async_main())
