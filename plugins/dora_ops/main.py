from __future__ import annotations

from pathlib import Path

try:
    from ncatbot.plugin import NcatBotPlugin
    from ncatbot.core import registrar
except Exception:  # pragma: no cover - lets core tests run without NcatBot internals.
    NcatBotPlugin = object  # type: ignore[misc,assignment]
    registrar = None  # type: ignore[assignment]

from dora_ops.runtime import DoraOpsRuntime


class DoraOpsPlugin(NcatBotPlugin):  # type: ignore[misc,valid-type]
    name = "dora_ops"
    version = "0.1.0"

    async def on_load(self) -> None:
        self.runtime = await DoraOpsRuntime.create(Path("config.yaml"))

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
            result = await self.runtime.handle_admin_text(text, user_id=user_id, group_id=group_id)
            if result:
                await event.reply(text=result)

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
            result = await self.runtime.handle_admin_text(text, user_id=user_id, group_id=group_id)
            if result:
                await self.api.post_group_msg(group_id, text=result)

    @staticmethod
    def _extract_text(msg) -> str:
        chunks: list[str] = []
        for item in getattr(msg, "message", []) or []:
            if item.get("type") == "text":
                chunks.append(item.get("data", {}).get("text", ""))
        return "".join(chunks)
