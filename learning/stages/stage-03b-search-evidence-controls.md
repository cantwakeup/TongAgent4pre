# Stage 03B：搜索质量与证据控制强化

## 1. 阶段目标

Stage 03A 已经把 research plan、SQ 状态和 coverage 变成代码拥有的显式状态，但 BIGAI 的两次真实运行暴露出四个执行层问题：

1. SQ1 可以耗尽整个 plan 的搜索和抓取额度，SQ2 尚未开始就没有工具预算。
2. DuckDuckGo 返回空结果或低相关噪声时，没有自动启用备用搜索引擎。
3. 权威官网中的短落地页会被固定 500 字符门槛一刀切拒绝。
4. researcher 可能在自然语言中局部重编号，把一个 URL 错称为已经存在的 `[S#]`。

本阶段只修复这四项控制问题。正文片段、Claim-Evidence 映射、语义支持度和冲突图仍不属于本阶段已完成能力。

## 2. 每个 SQ 的保留预算

`ResearchBudget.configure_subquestions` 将 plan 的总搜索/抓取上限确定性切片，`activate_subquestion` 决定当前哪个 SQ 可以消费自己的份额。

medium 两个 SQ 的例子：

```text
plan: 4 search / 6 fetch
  SQ1: 2 search / 3 fetch
  SQ2: 2 search / 3 fetch
```

`reserve_search` 和 `reserve_fetch` 同时检查 plan 总上限与当前 SQ 上限。拒绝原因区分为：

- `plan_budget_exceeded`
- `subquestion_budget_exceeded`
- `no_active_subquestion`

预算快照新增 `active_subquestion_id`、`subquestion_limits` 和 `subquestion_usage`，所以中断恢复后不会重新发放已经消费的额度。进入 report 阶段后激活范围为 `None`，网络工具不再可用。

## 3. 搜索相关性与 Bing 回退

搜索层先抓取 DuckDuckGo，再以可复现的词法规则评分：

- `site:` host 命中
- 引号短语命中
- 英文关键词覆盖
- 中文连续双字 overlap

DuckDuckGo 失败、为空，或达到阈值的结果少于两条时，自动请求 Bing RSS。合并结果会去重并携带：

- `engine`
- `relevance_score`
- `fallback_reason`
- `search_quality`
- `relevant_results`

这些字段用于路由和审计，不把词法重合冒充语义证据。真实 smoke 中，宽泛 BIGAI 查询触发了 `duckduckgo_empty -> bing`；`site:bigai.ai 北京通用人工智能研究院` 从 Bing 返回了 BIGAI 官网与科研部门页。

## 4. 自适应短页证据

页面证据分为两档：

| 可见正文长度 | 处理 |
| ---: | --- |
| `< 300` | `insufficient_content` |
| `300–499` | 仅当同 host 已有一条 `>= 500` 的完整证据时，以 `limited` 入账 |
| `>= 500` | 以 `full` 入账 |

`limited` 来源会保存 `quality_reason`，prompt 要求只用于窄事实并披露限制。未锚定的其他域短页仍会被拒绝。

BIGAI smoke 的机器结果：

```text
S1  /about/               1488 chars  full
S2  /tongprogram-2026/     430 chars  limited
```

完整模型复跑中编号随实际抓取顺序变为：

```text
S1  /                       1179 chars  full
S2  /about/                 1488 chars  full
S3  /tongprogram-2026/       430 chars  limited
```

## 5. Canonical Source Ledger

`get_source_ledger` 向 parent、single agent 和 researcher 返回同一份成功来源账本。账本是 `[S#]`、标题、URL 和证据质量的唯一编号来源。

代码执行以下约束：

1. `update_subquestion` 拒绝账本中不存在的 ID。
2. covered/blocked 携带证据时，至少一条必须来自当前研究步骤。
3. final synthesis 直接注入完整 canonical ledger。
4. CLI 拒绝报告中的未知 ID。
5. CLI 拒绝未挂载到当前 plan 的旧 thread 来源。
6. Sources 映射必须在同一行严格绑定 canonical ID、标题和 URL；交换 S1/S2 的 URL 不能再通过。

这能阻止本地重编号和映射错位，但尚不能自动判断“某个事实是否真的被该页面正文支持”。

## 6. 验证证据

离线回归：

```text
Ran 39 tests ... OK
All checks passed!
14 files already formatted
```

新增覆盖包括：

- SQ 预算切片和 checkpoint 恢复
- DDG 低相关触发 Bing；高相关时不触发
- DDG 与 Bing 同时失败时返回可审计空结果
- 中文实体噪声识别
- 同域完整证据锚定 430 字符短页
- 未锚定短页拒绝
- canonical ledger 工具与当前步骤证据约束
- 交换 URL、错误标题的逐行来源映射拒绝

BIGAI 完整复跑配置：

```text
effort=medium
mode=multi
model=gpt-5.4-nano
worker_model=deepseek-v4-flash
thread=bigai-stage03b-verify-20260717-01
```

结果：

```text
validation.status = passed
plan.status       = completed
coverage          = 1.0
SQ1 usage         = 2 search / 3 fetch
SQ2 usage         = 1 search / 3 fetch
sources           = S1 homepage, S2 about, S3 TongProgram-2026
```

报告将 TongProgram-2026 正确映射为 `[S3]`，没有再次把 homepage 的 `[S2]` 错称为 TongProgram。

## 7. 已知限制

- Bing RSS 对未索引的新页面仍可能返回低相关结果；回退保证多一个检索路径，不保证一定命中。
- `limited` 判定依赖同 host 的完整证据先被抓取，当前没有对先失败、后锚定的短页做自动晋升。
- coverage 仍是结构覆盖率。BIGAI 复跑虽为 1.0，但报告明确披露团队、论文、合作、开源和通计划细节仍有证据缺口。
- 来源标题和 URL 可以做确定性绑定；claim 与正文片段之间的语义支持仍由模型判断。
- token 统计仍不含结构化 planner 和部分 nested subagent 调用，也没有供应商账单金额。

下一步应在这些稳定控制之上建立正文片段级 Evidence Graph、Claim-Evidence-URL 映射、来源可信度和冲突对象，而不是继续依赖整页 `[S#]`。
