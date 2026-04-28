#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="${DORA_BOT_SESSION:-dora_bot}"
LOG_DIR="$ROOT_DIR/logs"
LOG_FILE="$LOG_DIR/ncatbot.log"

mkdir -p "$LOG_DIR"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required but was not found." >&2
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Dora bot is already running in tmux session: $SESSION_NAME"
  exit 0
fi

cd "$ROOT_DIR"

if [[ ! -f dora-bot.yaml ]]; then
  cp dora-bot.example.yaml dora-bot.yaml
  echo "Created dora-bot.yaml from dora-bot.example.yaml. Edit admin/bot settings before production use."
fi

tmux new-session -d -s "$SESSION_NAME" \
  "cd '$ROOT_DIR' && uv run ncatbot run --plugins-dir plugins --non-interactive 2>&1 | tee -a '$LOG_FILE'"

echo "Dora bot started."
echo "tmux session: $SESSION_NAME"
echo "log: $LOG_FILE"
echo "attach: tmux attach -t $SESSION_NAME"
