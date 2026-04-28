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
  acknowledge_feedback: true
  daily_group_analysis_limit: 3
  daily_user_analysis_limit: 1
  auto_create_analysis_jobs: false
```

With `enabled_group_ids: []`, every group is allowed. For production, fill explicit group ids. The current group-chat path is conservative: Dora SSR/YueScript-related feedback is recorded and acknowledged, unrelated messages are ignored, and deep repository analysis waits for admin confirmation unless later enabled explicitly.

Useful commands:

```bash
uv run dora-bot --config dora-bot.yaml admin '/test ping' --user-id 123456
uv run dora-bot --config dora-bot.yaml admin '/test classify Dora SSR Web IDE 无法刷新' --user-id 123456
uv run dora-bot --config dora-bot.yaml admin '/test repo-check Dora-SSR' --user-id 123456
uv run dora-bot --config dora-bot.yaml admin '/test daily-summary --dry-run' --user-id 123456
```

The first implemented NcatBot surface is the admin test console. These commands only run in private chat with an admin account:

- `/test ping`
- `/test classify <文本>`
- `/test feedback <文本>`
- `/test repo-check Dora-SSR|YueScript`
- `/test tmux`
- `/test opencode Dora-SSR|YueScript`
- `/test daily-summary --dry-run`
- `/test job-status --include-test`
- `/approvals`
- `/approve feedback <id>`
- `/reject feedback <id>`

`/test tmux`, `/test opencode ...`, and `/test daily-summary --progress` create asynchronous jobs under `jobs/`. `/test daily-summary --progress` tests the yesterday progress analysis path for all configured repositories. `/test job-status --include-test` reconciles finished jobs and shows the output summary when available.

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
```

At the scheduled time the bot creates one opencode job per repository. Each job runs in the repository directory, executes `git pull -f origin <branch>`, then asks opencode to analyze yesterday's changes.

Manual test trigger:

```text
/test daily-summary --progress
```

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
