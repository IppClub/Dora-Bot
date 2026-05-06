from __future__ import annotations

from dataclasses import dataclass

from .config import BotConfig


@dataclass(frozen=True)
class WelcomeMember:
    group_id: int
    user_id: int
    name: str = ""
    operator_id: int | None = None


class _TemplateValues(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class WelcomeService:
    def __init__(self, config: BotConfig):
        self.config = config

    def render(self, member: WelcomeMember) -> str | None:
        if not self.is_enabled_for_group(member.group_id):
            return None
        name = member.name.strip() or str(member.user_id)
        values = _TemplateValues(
            {
                "name": name,
                "nickname": name,
                "user_id": str(member.user_id),
                "group_id": str(member.group_id),
                "operator_id": str(member.operator_id or ""),
            }
        )
        text = self.config.welcome.message.format_map(values).strip()
        return text or None

    def is_enabled_for_group(self, group_id: int) -> bool:
        welcome = self.config.welcome
        if not welcome.enabled:
            return False
        if welcome.enabled_group_ids:
            return group_id in welcome.enabled_group_ids
        enabled_group_ids = self.config.group_chat.enabled_group_ids
        return not enabled_group_ids or group_id in enabled_group_ids
