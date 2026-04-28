from __future__ import annotations

from pathlib import Path

try:
    from ncatbot.plugin import NcatBotPlugin
    from ncatbot.core import registrar
except Exception:  # pragma: no cover - lets core tests run without NcatBot internals.
    NcatBotPlugin = object  # type: ignore[misc,assignment]
    registrar = None  # type: ignore[assignment]

from dora_ops.runtime import DoraOpsRuntime
from dora_ops.group_chat import GroupMessageInput


class DoraOpsPlugin(NcatBotPlugin):  # type: ignore[misc,valid-type]
    name = "dora_ops"
    version = "0.1.0"

    async def on_load(self) -> None:
        self.runtime = await DoraOpsRuntime.create(Path("dora-bot.yaml"))
        if hasattr(self, "add_scheduled_task"):
            self.add_scheduled_task(
                "daily_progress_report",
                self.runtime.config.scheduler.daily_summary_time,
                callback=self.daily_progress_report,
            )

    async def daily_progress_report(self) -> None:
        await self.runtime.summaries.create_yesterday_progress_jobs(self.runtime.jobs)

    if registrar is not None:

        @registrar.qq.on_private_message()
        async def on_private_message(self, event) -> None:
            text = getattr(event, "raw_message", "") or getattr(event, "text", "")
            user_id = int(getattr(event, "user_id", 0))
            result = await self.runtime.handle_admin_text(text, user_id=user_id)
            if result:
                await event.reply(text=result)

        @registrar.qq.on_group_message()
        async def on_group_message(self, event) -> None:
            text = getattr(event, "raw_message", "") or self._extract_text(event)
            user_id = int(getattr(getattr(event, "sender", None), "user_id", 0) or getattr(event, "user_id", 0))
            group_id = int(getattr(event, "group_id", 0))
            admin_result = await self.runtime.handle_admin_text(text, user_id=user_id, group_id=group_id)
            if admin_result:
                await self._send_group_reply(group_id, admin_result)
                return
            group_result = await self.runtime.handle_group_message(
                GroupMessageInput(
                    group_id=group_id,
                    user_id=user_id,
                    nickname=str(getattr(getattr(event, "sender", None), "nickname", "")),
                    text=text,
                    mentions_bot=self._mentions_bot(event),
                )
            )
            if group_result and group_result.reply:
                await self._send_group_reply(group_id, group_result.reply, group_result.mention_admin_id)

    else:

        async def on_private_message(self, msg) -> None:
            text = getattr(msg, "raw_message", "") or getattr(msg, "text", "")
            user_id = int(getattr(msg, "user_id", 0))
            result = await self.runtime.handle_admin_text(text, user_id=user_id)
            if result:
                await self.api.post_private_msg(user_id, text=result)

        async def on_group_message(self, msg) -> None:
            text = getattr(msg, "raw_message", "") or self._extract_text(msg)
            user_id = int(getattr(getattr(msg, "sender", None), "user_id", 0) or getattr(msg, "user_id", 0))
            group_id = int(getattr(msg, "group_id", 0))
            admin_result = await self.runtime.handle_admin_text(text, user_id=user_id, group_id=group_id)
            if admin_result:
                await self._send_group_reply(group_id, admin_result)
                return
            group_result = await self.runtime.handle_group_message(
                GroupMessageInput(
                    group_id=group_id,
                    user_id=user_id,
                    nickname=str(getattr(getattr(msg, "sender", None), "nickname", "")),
                    text=text,
                    mentions_bot=self._mentions_bot(msg),
                )
            )
            if group_result and group_result.reply:
                await self._send_group_reply(group_id, group_result.reply, group_result.mention_admin_id)

    @staticmethod
    def _extract_text(msg) -> str:
        chunks: list[str] = []
        for item in getattr(msg, "message", []) or []:
            if item.get("type") == "text":
                chunks.append(item.get("data", {}).get("text", ""))
        return "".join(chunks)

    @staticmethod
    def _mentions_bot(msg) -> bool:
        for item in getattr(msg, "message", []) or []:
            if item.get("type") == "at":
                return True
        return False

    async def _send_group_reply(self, group_id: int, text: str, at_user_id: int | None = None) -> None:
        if at_user_id is not None:
            await self.api.post_group_msg(group_id, text=text, at=at_user_id)
            return
        await self.api.post_group_msg(group_id, text=text)
