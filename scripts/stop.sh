#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${DORA_BOT_SESSION:-dora_bot}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required but was not found." >&2
  exit 1
fi

if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Dora bot is not running: $SESSION_NAME"
  exit 0
fi

tmux send-keys -t "$SESSION_NAME" C-c
sleep 2

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  tmux kill-session -t "$SESSION_NAME"
fi

echo "Dora bot stopped: $SESSION_NAME"
