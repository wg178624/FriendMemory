from __future__ import annotations

import os
import shlex
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


def load_dotenv(path: str | Path | None = None, *, override: bool = False) -> dict[str, str]:
    env_path = Path(path) if path is not None else DEFAULT_ENV_PATH
    if not env_path.exists():
        return {}

    loaded: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


def _parse_env_line(raw_line: str) -> tuple[str, str] | None:
    line = raw_line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export ") :].strip()
    if "=" not in line:
        return None

    key, value = line.split("=", 1)
    key = key.strip()
    if not key or not _valid_env_key(key):
        return None

    value = value.strip()
    if not value:
        return key, ""

    try:
        parts = shlex.split(value, comments=True, posix=True)
    except ValueError:
        return key, value
    if not parts:
        return key, ""
    return key, " ".join(parts)


def _valid_env_key(key: str) -> bool:
    return key.replace("_", "").isalnum() and not key[0].isdigit()
