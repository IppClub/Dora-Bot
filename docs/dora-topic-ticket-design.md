# Dora Topic：群聊工单与话题归档设计

## 1. 目标

让 Dora Bot 在 QQ 群内把明确的 issue 或 feature 设计沉淀为可追踪的话题工单，并将带有工单编号的后续讨论归集、分析和持久化到 AtomGit。

- 创建入口：群友 `@多萝` 后提出明确的项目 issue 或 feature 设计。
- 工单编号：`ds1`、`ds2`……；全局递增且唯一，大小写不敏感。
- 归集入口：后续消息含独立编号（例如 `ds1`、`DS1`）即可关联；不必重复 @ Bot。
- 归档结果：Bot 保存包含用户与多萝回复的完整讨论线程，并在 T+1 审阅后更新独立的分析设计文稿，提交并推送到 AtomGit。
- 目标边界：此功能记录和整理讨论，不替代 GitHub/AtomGit 的 Issue，也不在未批准时修改源码仓库。

## 2. 交互约定

### 2.1 创建

```text
群友：@多萝 Web IDE 需要支持离线保存草稿，避免掌机断网时丢失编辑内容。
多萝：已建工单 ds1：Web IDE 离线草稿保存。后续讨论带 ds1 即可归集。
```

只有同时满足以下条件才创建工单：

1. 消息或本轮防抖缓冲中明确 @ Bot；
2. 分类器判定为 Dora SSR/YueScript 相关的 `feedback` 或 `feature_design`；
3. LLM 规划器确认有可描述的目标、问题或方案，而非一句闲聊或重复讨论。

当信息不足时，不创建编号，只追问最关键的缺失信息（现象/目标、平台或版本、预期行为）。

### 2.2 归集与反馈

每条命中工单编号的群消息都会保存原文、作者、时间和所属群，随后由多萝参与讨论；Bot 回复同样写入该工单线程。每次回复最多使用最近 20 轮（至多 40 条用户/Bot 消息）作为上下文，不混入其他工单或全群历史。即时阶段不提交 Git：

```text
ds1 已归集，明天会基于今天的讨论生成设计分析稿。
```

消息中出现多个编号时，保存到每一个匹配工单。普通文本中的子串不匹配，例如 `ads123` 不应命中 `ds1`。

### 2.3 T+1 日审阅

调度器每天在 `review_time` 执行：对每个开放工单，审阅**昨天**带编号的新增讨论，结合上一版设计稿更新 `designs/ds1.md`。例如 7 月 17 日的讨论由 7 月 18 日生成或修订设计稿。默认由 Bot 在目标项目的本地仓库中启动 OpenCode：OpenCode 读取归档仓库的 `topics/ds1.md`，分析真实源码后只写回归档仓库的 `designs/ds1.md`；Bot 负责校验、提交和推送。

仅在以下条件同时成立时更新文稿：

1. 昨天至少有一条该工单的新讨论；
2. 昨天最新消息的 `id` 大于工单的 `last_reviewed_message_id`；
3. 当天尚未成功完成同一工单的日审阅。

没有新讨论时直接跳过，不调用 LLM、不改文件、不产生空提交。文稿 front matter 同时写入“讨论更新至”和“文稿更新于”；前者来自最后一条已纳入分析的群消息日期，后者来自实际生成时间，不能把单纯的定时任务运行误写成讨论更新。

### 2.4 状态与人工控制

工单状态为 `open`、`paused`、`closed`。管理员私聊命令：

```text
/topic list [open|paused|closed]
/topic show ds1
/topic close ds1 [结论]
/topic reopen ds1
/topic review ds1
```

关闭后的编号仍保存讨论，但默认不再参与日审阅；管理员可使用 `/topic review ds1` 强制生成一次最终分析稿，Bot 会先私聊确认“正在生成”，任务完成后再私聊返回文稿链接；添加 `/topic review ds1 --notify-group` 才会把链接同步到该工单来源群。定时 T+1 审阅成功后会自动向对应来源群推送链接。话题设计作业默认最多运行 20 分钟。普通群成员不能用命令更改状态。

## 3. 配置

在 `dora-bot.yaml` 新增独立配置，不能复用 `repositories.*`：前者是源码分析目标，后者是 Bot 可写的归档目标。

```yaml
topic_archive:
  enabled: true
  repository:
    local_path: /srv/dora-topic
    remote: git@atomgit.com:ippclub/dora-topic.git
    branch: main
  docs_dir: topics
  design_dir: designs
  review_time: "08:30"
  timezone: Asia/Shanghai
  daily_review_24h_limit: 60
  design_generator: opencode # 可为 fallback，用于没有 OpenCode 的测试/降级环境
  public_url: https://atomgit.com/ippclub/dora-topic
```

启动时必须验证：`local_path/.git` 存在、当前分支与 `branch` 一致、`origin` URL 与配置一致且工作区干净。任何一项不满足时禁用同步并通知管理员，绝不能在未知工作区执行 `git pull`、`commit` 或 `push`。

服务器应通过专用 SSH deploy key 获得该仓库的写权限；密钥不进入 YAML、数据库、作业 prompt 或日志。

## 4. 数据模型

在现有 SQLite `Storage` 中增加三张表，并为旧数据库采用 `create table if not exists` 迁移。

```sql
create table topic_ticket(
  id integer primary key autoincrement,
  group_id integer not null,
  sequence integer not null unique,
  code text not null unique,
  project text,
  kind text not null,
  title text not null,
  original_text text not null,
  status text not null default 'open',
  created_by integer not null,
  created_at integer not null,
  closed_at integer,
  close_note text,
  last_message_at integer,
  last_reviewed_message_id integer,
  last_reviewed_for_date text,
  last_design_updated_at integer,
  last_review_commit text
);

create table topic_message(
  id integer primary key autoincrement,
  ticket_id integer not null references topic_ticket(id),
  group_id integer not null,
  user_id integer not null,
  nickname text not null default '',
  role text not null check(role in ('user', 'assistant')),
  content text not null,
  source_message_at integer not null,
  created_at integer not null
);

create table topic_sync(
  id integer primary key autoincrement,
  ticket_id integer not null references topic_ticket(id),
  kind text not null default 'daily_review',
  review_date text,
  from_message_id integer,
  to_message_id integer,
  artifact_path text,
  status text not null,
  commit_sha text,
  error text,
  created_at integer not null,
  finished_at integer
);
```

`sequence` 在所有群中从 1 递增，显示码由 `ds{sequence}` 生成；数据库始终保存小写 `code`。`topic_message` 是权威讨论线程记录，`role` 区分用户与多萝。`topics/*.md` 保存工单元数据、原始诉求与完整讨论线程，`designs/*.md` 是由 T+1 审阅生成、可由数据库消息重建的设计分析稿。

## 5. 服务划分

新增以下模块，保留 `GroupMessageService` 负责入口路由而不是承载全部业务：

- `topic_classifier.py`：判定 `issue`、`feature_design`、`not_topic`，并输出标题、项目和置信度。
- `topic_service.py`：创建工单、抽取编号、保存用户/Bot 线程消息、权限和状态判断。
- `topic_review.py`：选择昨天有新增讨论的工单，防重、调度和失败重试；默认提交 OpenCode 作业。
- `topic_renderer.py`：在 `fallback` 模式调用 LLM 生成结构化的分析设计内容，渲染确定性的 Markdown。
- `topic_archive.py`：唯一允许触碰归档 Git 工作区的组件，串行写入 topic、校验 OpenCode 仅写入目标设计稿、提交与推送。
- `prompts.py`：增加工单讨论与 T+1 设计审阅 prompt；讨论回复使用同一工单最近 20 轮线程。OpenCode 日审 prompt 只给出对应 `topics/dsN.md` 的绝对路径与 `designs/dsN.md` 输出路径，要求基于本地项目仓库分析，不混入全群 `chat_message`。

在 `GroupMessageService.handle()` 中的顺序：

1. 写入现有群聊历史；
2. 优先检测已存在的 `dsN`，命中后交给 `TopicService.record_references()`；
3. 未命中时，仅在 `mentions_bot` 为真时执行工单创建分类；
4. 未创建工单的现有反馈/仓库分析逻辑维持不变。

这保证一个带 `ds1` 的后续讨论不会被再次创建为普通 `feedback`，也不会污染其他工单的上下文。

## 6. Markdown 与 LLM 输出

工单索引固定为 `topics/ds1.md`，记录元数据、原始诉求、设计稿链接和完整的用户—多萝讨论线程。分析设计稿固定为 `designs/ds1.md`，不按标题命名，避免重命名破坏链接。其格式：

```markdown
# ds1：Web IDE 离线草稿保存

- 状态：open
- 项目：Dora-SSR
- 创建于：2026-07-17
- 讨论更新至：2026-07-17
- 文稿更新于：2026-07-18 08:30 +08:00
- 本次审阅范围：消息 #12 - #19

## 问题定义与目标

## 方案与取舍

## 建议的实现拆分

## 风险与验证

## 待确认问题

## 本次讨论结论

- <只总结有证据的新增结论；保留与消息编号的对应关系>

## 关联

- 工单索引：`topics/ds1.md`
```

LLM 必须返回受 schema 约束的 JSON（问题定义、结论、方案、取舍、实现拆分、风险、待决项、消息范围），由 renderer 写入模板；模型不能直接执行 shell 或 Git 命令。没有明确证据的项必须进入“待确认问题”，不能写成已确认结论。旧稿中未被新讨论推翻的结论必须保留，避免日审阅把已完成的设计反复缩短或覆盖。

## 7. 同步与 Git 安全

日审阅由 `TopicArchive` 的单一 `asyncio.Lock` 串行化。每一次成功审阅：

1. 查询昨天的同工单消息，并检查 `last_reviewed_message_id` 与 `last_reviewed_for_date`；无新消息或已成功审阅则退出；
2. 验证归档仓库的 origin、分支和工作区；干净时执行 `git fetch origin main` 与快进式 `git pull --ff-only origin main`；
3. 写入完整的 `topics/dsN.md`，在目标项目本地仓库启动 OpenCode。作业可读项目源码和 topic 文件，但 prompt 禁止修改源码或执行 Git 写入，只允许写入指定的 `designs/dsN.md`；
4. OpenCode 成功后，验证归档工作区仅变更 `topics/dsN.md` 与 `designs/dsN.md`，且设计文档非空；
5. `git add -- topics/dsN.md designs/dsN.md`，只允许这两个目标路径进入暂存区；
6. `git commit -m "docs(topic): review dsN for YYYY-MM-DD"`；
7. `git push origin main`；
8. 仅在推送成功后更新 `last_reviewed_message_id`、`last_reviewed_for_date`、`last_design_updated_at` 和 `last_review_commit`。

若拉取冲突、工作区脏、push 失败或 OpenCode/LLM 失败，保留原始消息、写入 `topic_sync.error`、向管理员告警；不更新审阅游标，因此下次可安全重试。不重置、不强推、不覆盖远端文档。

## 8. 实施阶段与验收

### Phase 1：本地工单闭环

- 配置模型、SQLite 迁移、`TopicService`、编号解析及管理员只读命令。
- 单元测试：创建 `ds1`、跨群全局递增、编号边界匹配、关闭后不自动归集。

### Phase 2：T+1 设计审阅与渲染

- 受约束 OpenCode prompt、归档路径白名单、在目标项目仓库执行的作业编排，以及 `fallback` 的确定性 Markdown renderer。
- 测试：OpenCode 收到正确的项目路径与 topic 文档、只能写指定设计稿、昨天没有新消息时零作业/零 Git 写入、同一天重复运行不更新、重试不重复摘要。

### Phase 3：AtomGit 归档

- `TopicArchive`、干净工作区校验、串行 push、失败恢复和管理员简短回执。
- 集成测试使用临时 bare Git remote；覆盖 `--ff-only`、push 失败、只暂存 `topics/dsN.md designs/dsN.md`、重启后恢复未完成审阅。

### Phase 4：可观测性与上线

- `/topic list/show/review/close/reopen`、管理员通知、每日审阅配额、结构化日志和指标。
- 在测试群验证：创建 `ds1` → 两人多轮讨论 → 次日生成设计稿并推送 → 无新讨论的次日不提交 → AtomGit 可访问 → 关闭工单 → 管理员最终审阅。

上线验收标准是：每条带编号的有效讨论均可在数据库与 AtomGit 文档中追溯；任何 Git/LLM 失败都不丢失原始消息、不污染源码仓库、不产生未声明的远端写入。
