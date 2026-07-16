# Stage 03A：显式研究计划与可恢复状态图

日期：2026-07-16

状态：已完成，可复现

前一阶段：`stage-02` 分级、多 Agent、可恢复研究闭环

## 一句话成果

TongAgent 不再只依赖 prompt 隐式规划：系统现在会生成机器可读的研究计划，按稳定 `SQ#` 子问题执行 `plan -> select -> research -> evaluate -> report` 状态图，并把计划、结构覆盖度、预算和事件保存到 SQLite checkpoint 与审计文件。

## 1. 本阶段解决的问题

Stage 02 能恢复消息，但无法回答：

- 当前研究计划是什么？
- 哪个子问题已完成、受阻或尚未开始？
- 覆盖度由什么规则计算？
- 进程中断后应从哪一步继续？
- 模型是否跳过了未完成工作？

Stage 03A 将这些信息从 prompt 和自然语言中提取为代码拥有的显式状态。

## 2. 已完成能力

| 能力 | 代码行为 | 验证证据 |
| --- | --- | --- |
| 结构化规划 | 低成本主模型输出 `PlanDraft`；失败时确定性 fallback | 真实模型规划与纯函数测试 |
| 稳定子问题 | 代码分配 `SQ1...SQn`，模型不能设置运行状态 | 计划规范化测试 |
| 状态转换 | `pending -> researching -> covered/blocked`；依赖 blocked 会级联阻塞下游 | 状态转换与依赖测试 |
| 结构覆盖度 | `covered / all subquestions`，不信任模型自报分数 | 覆盖度测试 |
| 显式外层图 | `plan -> select -> research -> evaluate -> report -> finish` | 离线完整图测试 |
| 尝试上限 | 每个子问题最多两次；达到上限转为 blocked | 停滞与防无限循环测试 |
| 节点级恢复 | CLI 依据 `checkpoint.next` 用 `invoke(None)` 继续中断节点 | CLI 与跨连接恢复测试 |
| 旧状态兼容 | message-only checkpoint 状态形状可由新图补齐字段 | schema-level 兼容测试 |
| 预算恢复 | 未完成 plan 恢复计数；新 plan 重置额度但延续 thread 来源 ID | 跨连接预算测试 |
| 状态工具 | 只更新 active SQ，并校验当前步骤 `[S#]` 属于成功来源账本 | 工具校验测试与真实 trace |
| 新审计产物 | 保存 `plan.json` 与有序 `events.jsonl` | artifact 测试与真实运行 |

## 3. 架构

```text
START
  |
  v
plan -- 新问题 --> 结构化 planner / deterministic fallback
  |
  v
select ----------> 选择 researching 或 dependency-ready pending SQ
  |
  v
prepare_research -> 注入 [RESEARCH STEP]
  |
  v
Deep Agent ------> single: 直接搜索
  |                multi: task(researcher)
  |                get_research_plan / update_subquestion
  v
evaluate --------> 代码计算结构 coverage、预算和尝试次数
  |                         |
  | unfinished + budget     | terminal / cycle limit
  +-----------> select      v
                       prepare_report
                              |
                              v
                       [FINAL SYNTHESIS]
                              |
                              v
                           finish
                              |
                             END
```

checkpoint 只挂在外层研究图。内层 Deep Agent 显式设置 `checkpointer=False`；外层包装节点在 inner graph 返回时把普通预算 snapshot 与节点结果一起提交，因此不会与同一 `thread_id` 竞争 SQLite 状态，也不会在 `research_agent -> evaluate` 的正常节点边界丢失来源编号。

## 4. 状态契约

`TongAgentState` 在 Deep Agents 消息状态之外保存：

```text
research_topic
research_plan
research_plan_history
research_events
active_subquestion_id
active_source_ids_before
budget_state
workflow_phase
research_cycles
max_research_cycles
```

所有新增字段都是 JSON 可序列化的字符串、数字、列表和字典。包含 `Lock` 的进程内 `ResearchBudget` 不直接进入 checkpoint，只保存其普通 snapshot。

每个子问题保存：

```text
id / question / rationale / depends_on
status / attempts / max_attempts
evidence_source_ids / note
```

`covered` 必须由当前 active 子问题提交，携带成功来源账本中真实存在的 `[S#]`，并且至少一个 ID 来自当前研究步骤；`blocked` 必须保存原因。依赖项 blocked 时，下游会级联 blocked，已 covered 的工作不会再次被选择。

这里的 coverage 明确定义为“结构覆盖率”：代码能证明每个 SQ 是否走完合法状态转换、是否引用了本步骤真实抓取的来源 ID，但尚不能自动证明网页正文在语义上支持该问题。后者需要后续 Evidence Graph 和 Claim-Evidence 映射。

## 5. 分级计划宽度

显式计划宽度现在也是资源策略的一部分：

| effort | 最多子问题 | 搜索/抓取上限 |
| --- | ---: | ---: |
| low | 1 | 2 / 3 |
| medium | 2 | 4 / 6 |
| high | 4 | 8 / 9 |
| xhigh | 5 | 12 / 14 |

真实开发中，最初允许 low 生成两个子问题。第一个子问题耗尽两次搜索后，第二个必然 blocked。该结果促使计划宽度与工具预算绑定，而不是把“问题分得越细”误认为质量越高。

验收所需搜索数也从固定“两次”改为一个 plan-level 聚合下限：`min(max_searches, subquestion_count)`。它只约束总搜索次数，不伪装成 per-SQ 查询追踪；真正的 per-SQ 查询、token 和时延归因留给后续 telemetry 阶段。

## 6. checkpoint 语义

- 首次运行创建计划并保存 `plan_created`。
- 子问题选中后，外层图在进入 `research_agent` 前形成 checkpoint。
- inner graph 返回时，包装节点把最新预算与来源 snapshot 一并提交到外层 checkpoint。
- CLI 读取 `checkpoint.next`；若仍有 pending node，先以 `invoke(None)` 精确继续，不把命令行 topic 错当成新的节点输入。
- 未完成计划恢复原计划、事件、预算计数和来源 ID，不重新规划。
- 已完成计划收到新问题时，旧计划进入 `research_plan_history`，新建当前计划。
- 预算按 plan 计数，来源目录按 thread 累积：新 plan 的 search/fetch 从零开始，但新来源继续使用下一个 `[S#]`，不会与历史消息碰撞。
- 只含 messages 的旧 checkpoint 状态形状会补齐新字段，不丢失历史消息；当前测试不宣称覆盖所有历史 Stage 02 二进制 fixture。

边界：当前保证的是子问题节点级恢复。若进程恰好死在一次网络工具内部，该工具节点可能重跑；后续阶段再处理工具级幂等、重试和原子证据写入。

## 7. 审计产物

在 Stage 02 文件之外新增：

| 文件 | 内容 |
| --- | --- |
| `output/plan.json` | thread、当前计划、子问题、覆盖度和预算快照 |
| `output/events.jsonl` | 有序的计划创建、选择、更新、评估和结束事件 |

事件在 checkpoint 中的列表是 canonical state；JSONL 是运行结束后的物化结果，因此不会把外部 append 文件误当成恢复真相。

## 8. 离线验证

当前共 27 项离线测试，包含原 Stage 02 的 10 项以及 Stage 03A 新增或强化的 17 项。新增覆盖：

- planner 输出规范化与稳定 ID
- planner 失败后的有界 deterministic fallback
- 非法状态转换、假来源 ID、非 active SQ 和跨步骤旧来源拒绝
- blocked dependency 级联阻塞
- covered 项不重复研究
- 完整状态图顺序
- 尝试上限、partial 退出和 completed plan history
- SQLite 真正关闭、重开并从中断节点继续
- 真实 CLI 路径使用 `invoke(None)` 续跑 pending node
- message-only checkpoint 状态形状兼容
- `research_agent` 后、`evaluate` 前 fresh ledger 的预算与来源 ID 恢复
- completed thread 新 plan 的预算重置与来源 ID 连续性
- 当前 plan checkpointed AI messages 的 token 聚合
- `plan.json` / `events.jsonl` 格式和有序事件

```text
Ran 27 tests ... OK
ruff check ... All checks passed
ruff format --check ... already formatted
```

## 9. 真实路径验证

复现命令（使用测试阶段的 nano 模型；API 地址与 key 只放在本地 `.env`）：

```bash
.venv/bin/python search_agent.py \
  --effort low \
  --mode single \
  --model gpt-5.4-nano \
  --thread-id stage03a-smoke-your-unique-id \
  --no-stream \
  "LangGraph 是什么，适合哪些类型的工作流？"
```

最终研究/恢复实现的通过路径记录为 `stage03a-smoke-v4`。首次受限网络执行在 `research_agent` 节点失败，第二次遇到免费模型服务端 500，第三次从同一 pending node 继续并通过；因此这条记录同时覆盖了 deterministic planner fallback 和真实 CLI `invoke(None)` 恢复。token 聚合代码随后从同一 SQLite checkpoint 重算用量，数值经过独立复核，但原始 v4 `run.json` 生成时尚无该字段。

```text
topic: LangGraph 是什么，适合哪些类型的工作流？
effort: low
mode/topology: single
model: gpt-5.4-nano
planner: deterministic-fallback:APIConnectionError
resumed node: research_agent
subquestions: 1
searches: 1 / 2
fetches: 2 / 3
unique successful sources: 2
coverage: 1.0
plan status: completed
validation: passed
checkpointed model calls: 7
tokens: 33,287 input + 1,843 output = 35,130 total
cache-read tokens: 26,496
```

真实工具链：

```text
get_research_plan
-> web_search
-> fetch_url x2
-> update_subquestion(SQ1, covered, [S1, S2])
-> write_file
```

更早的 `stage03a-smoke-v3` 已验证 `planner: model` 正常路径；`v4` 则是在最终审查修复之后执行，选择保留更难的“planner 失败 fallback + 节点恢复”证据，而不是把服务端错误从开发记录中删掉。

脱敏后的机器可读快照已提交为 [`artifacts/stage-03a-smoke-v4.json`](artifacts/stage-03a-smoke-v4.json)；它保留模型档位、预算、公开来源、工具链、事件序列和恢复经过，不包含 API key、私有 base URL、网页正文或本机 checkpoint 路径。

真实状态事件为：

```text
plan_created
-> subquestion_selected
-> subquestion_updated
-> coverage_evaluated
-> report_requested
-> run_finished
```

## 10. 代码索引

- `research_state.py`：checkpoint schema
- `research_graph.py`：planner、转换规则、状态工具和外层图
- `telemetry.py`：plan/event artifact 物化
- `search_agent.py`：网络工具、Deep Agent 构建、CLI 与验收
- `tests/test_research_graph.py`：状态图与跨进程恢复
- `tests/test_research_artifacts.py`：审计产物

## 11. 已知限制与 Stage 03B

- 计划是显式的，但查询仍由模型在每个子问题内部生成。
- coverage 只是账本约束下的结构覆盖，证据相关性仍由模型判断。
- `evidence_source_ids` 仍指向整页来源，没有正文片段。
- 只有 URL 去重，尚无正文 hash、近似或语义去重。
- 没有结构化来源可信度和 Claim-Evidence-URL 图。
- blocked 只记录自然语言原因，尚未按错误类型执行恢复策略。
- `run.json` 已聚合 checkpointed AI messages 的 token，但 planner、部分 nested subagent、逐节点时延和真实账单费用仍没有完整 span。
- 输出目录仍是 last-run artifact，不支持同目录并发运行隔离。

本快照当时建议下一阶段直接建立 Evidence Graph。后续 BIGAI 真实运行先暴露出 SQ 预算饥饿、搜索回退、官方短页和来源编号错配四项更基础的问题，因此实际 [`stage-03b`](stage-03b-search-evidence-controls.md) 先完成这些控制层加固；正文片段、内容去重、来源可信度、Claim-Evidence-URL 映射和显式冲突对象顺延到后续阶段。

## 12. PPT 建议页

1. Stage 02 的恢复为什么仍只是“恢复对话”。
2. 外层研究图与内层 Deep Agent 的职责边界。
3. `SQ#` 状态机和代码计算的 coverage。
4. SQLite 中断恢复：为何 planner 不会再次运行。
5. low 两子问题失败如何反推计划宽度策略。
6. 真实通过轨迹：1 次搜索、2 个独立来源、coverage 1.0。
7. 从 Research Plan 走向 Evidence Graph 和自适应计算。
