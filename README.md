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
- `config.example.yaml`: starter configuration.

Copy `config.example.yaml` to `config.yaml` and fill in admin QQ ids before running the bot.

Useful commands:

```bash
uv run dora-bot --config config.yaml admin '/test ping' --user-id 123456
uv run dora-bot --config config.yaml admin '/test classify Dora SSR Web IDE 无法刷新' --user-id 123456
uv run dora-bot --config config.yaml admin '/test repo-check Dora-SSR' --user-id 123456
uv run dora-bot --config config.yaml admin '/test daily-summary --dry-run' --user-id 123456
```

The first implemented NcatBot surface is the admin test console:

- `/test ping`
- `/test classify <文本>`
- `/test feedback <文本>`
- `/test repo-check Dora-SSR|YueScript`
- `/test tmux`
- `/test opencode Dora-SSR|YueScript`
- `/test daily-summary --dry-run`
- `/test job-status --include-test`

`/test tmux` and `/test opencode ...` create asynchronous jobs under `jobs/`. Job output is reconciled by the core `JobManager`; the bot stores the tracked repository change first and fills in analysis results later.

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
