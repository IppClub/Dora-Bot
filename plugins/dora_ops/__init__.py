from __future__ import annotations

from pathlib import Path

_core_package_path = Path(__file__).resolve().parents[2] / "src" / "dora_ops"

if _core_package_path.is_dir():
    __path__.append(str(_core_package_path))  # type: ignore[name-defined]
