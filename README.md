# Dora Bot

Maintainer-focused QQ bot helpers for Dora SSR and YueScript.

The project is managed by `uv`:

```bash
uv sync
uv run pytest
```

The runtime is split into a testable core package and a thin NcatBot plugin adapter:

- `src/dora_ops/`: storage, repo tracking, tmux jobs, summaries, admin commands.
- `plugins/dora_ops/`: NcatBot plugin entrypoint.
- `dora-bot.example.yaml`: starter configuration.

Copy `dora-bot.example.yaml` to `dora-bot.yaml` and fill in admin QQ ids before running the bot.

Model configuration:

```yaml
llm:
  enabled: false
  max_context_messages: 20
  classifier:
    provider: openai-compatible
    base_url: https://api.deepseek.com
    api_key_env: DEEPSEEK_API_KEY
    model: deepseek-chat
  summarizer:
    provider: openai-compatible
    base_url: https://api.deepseek.com
    api_key_env: DEEPSEEK_API_KEY
    model: deepseek-chat
  chat:
    provider: openai-compatible
    base_url: https://api.deepseek.com
    api_key_env: DEEPSEEK_API_KEY
    model: deepseek-chat
```

Keep API keys in environment variables, not in `dora-bot.yaml`:

```bash
export DEEPSEEK_API_KEY='...'
```

`jobs.opencode_command` is separate from `llm`; it controls the tmux worker command used for repository analysis:

```yaml
jobs:
  opencode_command: opencode run
```

Group chat behavior:

```yaml
group_chat:
  enabled: true
  enabled_group_ids: []
  bot_aliases:
    - 多萝
    - Dora
  chat_enabled: true
  debounce_seconds: 60
  chat_cooldown_seconds: 120
  daily_group_analysis_limit: 3
  daily_user_analysis_limit: 1
  auto_create_analysis_jobs: false
```

With `enabled_group_ids: []`, every group is allowed. For production, fill explicit group ids. The current group-chat path is conservative: Dora SSR/YueScript-related feedback is recorded silently in the group and sent to admins by private message, unrelated messages are ignored, and deep repository analysis waits for admin confirmation unless later enabled explicitly. Group messages are buffered for `debounce_seconds` before one handling pass. When `llm.enabled` and `group_chat.chat_enabled` are true, group chat history is stored and the chat model may reply as Dora after being mentioned, or to high-confidence project questions after the cooldown window.

Useful commands:

```bash
uv run dora-bot --config dora-bot.yaml admin '/test ping' --user-id 123456
uv run dora-bot --config dora-bot.yaml admin '/test classify Dora SSR Web IDE 无法刷新' --user-id 123456
uv run dora-bot --config dora-bot.yaml admin '/test group-chat 123456789 Dora SSR Web IDE 无法刷新' --user-id 123456
uv run dora-bot --config dora-bot.yaml admin '/test repo-check Dora-SSR' --user-id 123456
uv run dora-bot --config dora-bot.yaml admin '/test daily-summary --dry-run' --user-id 123456
```

The first implemented NcatBot surface is the admin test console. These commands run in private chat with an admin account:

- `/test ping`
- `/test classify <文本>`
- `/test feedback <文本>`
- `/test group-chat <群号> <文本>`
- `/test repo-check Dora-SSR|YueScript`
- `/test tmux`
- `/test opencode Dora-SSR|YueScript`
- `/test daily-summary --dry-run`
- `/test job-status --include-test`
- `/approvals`
- `/approve feedback <id>`
- `/reject feedback <id>`

Admins can also send ordinary private-chat messages. Dora SSR/YueScript-related feedback is recorded, and messages that need repository analysis create a pending approval that can be approved with `/approve feedback <id>`. Private-chat and enabled group-chat history is stored and, when `llm.enabled: true`, the chat model receives the most recent `llm.max_context_messages` messages.

`/test group-chat ...` simulates the group-chat classifier, feedback recording, and approval request path from private chat. Pass the target group id explicitly.

`/test tmux`, `/test opencode ...`, and `/test daily-summary --progress` create asynchronous jobs under `jobs/`. `/test daily-summary --progress` tests the yesterday progress analysis path for all configured repositories and the plugin automatically tracks those jobs, then sends the final summaries back to the admin private chat. `/test job-status --include-test` can still be used to reconcile and inspect recent jobs manually.

When group chat records a feedback item that needs deeper repository analysis, it creates a pending approval. Admins approve or reject it in private chat:

```text
/approvals
/approve feedback 12
/reject feedback 12
```

Approving a feedback item creates a tmux/opencode job against the corresponding repository mirror.
The group acknowledgement mentions the first configured admin QQ via NcatBot's `post_group_msg(..., at=<qq>)` helper.

Daily project progress reports run at `scheduler.daily_summary_time`, which defaults to `08:00`. The report jobs use the configured working repositories directly:

```yaml
repositories:
  dora_ssr:
    local_path: /root/Workspace/Dora-SSR
  yuescript:
    local_path: /root/Workspace/YueScript

scheduler:
  daily_summary_time: "08:00"
  daily_summary_group_ids: [123456789]
```

At the scheduled time the bot creates one opencode job per repository. Each job runs in the repository directory, executes `git pull -f origin <branch>`, then asks opencode to analyze yesterday's changes. After all progress jobs succeed, fail, or time out, the plugin sends the final `昨日进展分析结果` to `scheduler.daily_summary_group_ids`. If that list is empty, it falls back to `group_chat.enabled_group_ids`.

Manual test trigger:

```text
/test daily-summary --progress
```

After the command creates jobs, the plugin polls them in the background and sends a final `昨日进展分析结果` message when all jobs succeed, fail, or time out.

NcatBot startup:

```bash
./scripts/start.sh
./scripts/stop.sh
```

The underlying command is:

```bash
uv run ncatbot run --plugins-dir plugins --non-interactive
```

For foreground development with hot reload:

```bash
uv run ncatbot dev --plugins-dir plugins
```
