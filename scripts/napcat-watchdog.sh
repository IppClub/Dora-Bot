#!/usr/bin/env bash
set -euo pipefail

WS_URL="${NAPCAT_WS_URL:-ws://127.0.0.1:3001}"
WS_TOKEN="${NAPCAT_WS_TOKEN:-napcat_ws}"
INTERVAL_SECONDS="${NAPCAT_WATCH_INTERVAL_SECONDS:-30}"
RESTART_CMD="${NAPCAT_RESTART_CMD:-}"
LOG_FILE="${NAPCAT_WATCHDOG_LOG:-/tmp/napcat-watchdog.log}"

if [[ -z "$RESTART_CMD" ]]; then
  echo "NAPCAT_RESTART_CMD is required, for example: napcat restart" >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG_FILE")"

check_ws() {
  NAPCAT_WS_URL="$WS_URL" NAPCAT_WS_TOKEN="$WS_TOKEN" python3 - <<'PY'
import asyncio
import json
import os
import sys
from urllib.parse import urlencode

try:
    import websockets
except ImportError:
    print("python package 'websockets' is required", file=sys.stderr)
    sys.exit(2)


async def main() -> int:
    url = os.environ["NAPCAT_WS_URL"]
    token = os.environ.get("NAPCAT_WS_TOKEN", "")
    if token:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}{urlencode({'access_token': token})}"

    try:
        async with websockets.connect(url, open_timeout=5, close_timeout=1) as ws:
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            data = json.loads(raw)
            if data.get("status") == "failed":
                print(data)
                return 1
            return 0
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1


raise SystemExit(asyncio.run(main()))
PY
}

while true; do
  if ! check_ws >>"$LOG_FILE" 2>&1; then
    {
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] NapCat WebSocket unhealthy; restarting"
      bash -lc "$RESTART_CMD"
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] restart command finished"
    } >>"$LOG_FILE" 2>&1
    sleep 10
  fi

  sleep "$INTERVAL_SECONDS"
done
