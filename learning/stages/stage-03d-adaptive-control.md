# Stage 03D：证据缺口驱动的有界 Adaptive Control

日期：2026-07-18

状态：03D 原型已完成；Benchmark Ready Sprint 阶段 A 指标语义加固后，
144 项离线回归通过，未重新调用付费模型

前一阶段：`stage-03c` exact quote、Evidence Graph 与严格报告映射

## 一句话成果

TongAgent 在保留 03C provenance 硬门槛的基础上，新增了一个不调用模型的
deterministic controller：`--strategy adaptive` 先给每个 SQ 一个最小
`1 search / 1 fetch` baseline，再根据 Claim、保守佐证来源组、相关搜索、工具失败和
Evidence 完整性缺口，有界、幂等地释放用户所选 effort 内的 reserve，并把每次
“继续、扩容、停止或结束”的原因写入 checkpoint 和审计产物。

## 1. 为什么需要 03D

Stage 03C 已能诚实拒绝缺少来源、Claim 闭包或 researcher 回执的运行，但它的
资源策略仍然是运行前一次性决定的：

- `mode=auto` 只解析初始 single/multi topology。
- effort 的全部搜索/抓取额度会在 plan 创建时固定分给所有 SQ。
- 网络失败、低相关搜索、内容过短和预算拒绝没有统一 attempt 序列。
- Evidence 缺口即使可由剩余全局预算恢复，也没有独立控制节点来决定是否追加。
- 恢复运行时缺少一个覆盖 strategy、模型、topology 和控制上限的统一身份。
- high/xhigh 的 reviewer 验收只看是否出现委派，不足以证明审稿成功且发生在
  最终写入之前。
- 所有运行写到同一个 last-run 目录会覆盖前一次审计证据。

03D 的目标不是“让模型自己觉得要多搜几次”，而是让代码从可审计状态做有界
决策。它仍然不会把一个不完整运行伪装成成功。

## 2. 审阅后保留的 03C 优点

03D 没有替换已经可靠的机制，而是在其外侧增加控制层：

| 保留项 | 继续成立的边界 |
| --- | --- |
| 显式 `ResearchPlan/SubQuestion/BudgetState` | controller 读取同一持久状态，不建立第二套隐藏 plan |
| exact quote 严格正文子串 | adaptive 只给额度，不放松 quote 或 Claim 规则 |
| `C#/E#/S#/X#` 图与 revision hash | 完整性错误优先 `fail_closed` |
| multi parent 只读 Evidence | 新证据仍只能由 researcher 登记 |
| 调用、provider、非空、相关与产证据分离 | attempt ledger 统一 search/fetch 的失败语义 |
| 来源与 revision 分离 | 分别记录正文 revision、hostname 与保守佐证来源组 |
| 当前 step、正确 SQ、成功 researcher 回执 | controller 不能绕过委派门槛 |
| 四个 H2、exact claim.text、deterministic Sources | adaptive 不改变报告协议 |
| 外层 checkpoint | control state、grant 和 attempts 一起恢复 |
| 诚实 partial | ceiling 到达后停止 SQ，最终验收仍可拒绝 partial plan |

## 3. 阶段边界

### 03D 已实现

- CLI 显式支持 `--strategy fixed|adaptive`。
- `--effort low|medium|high|xhigh` 始终是 plan 的硬 ceiling。
- adaptive baseline、全局 reserve 和单调 grant。
- `--max-escalations` 整体升级次数护栏。
- search/fetch 共用的结构化 attempt ledger。
- `evaluate -> adaptive_control` 确定性控制节点。
- Evidence 完整性、缺口、失败、无进展和预算信号。
- controller decision、reason codes、预算 before/after 和控制事件。
- checkpoint 恢复时的 policy fingerprint、plan/budget scope 对齐与幂等 grant。
- grant 每次每维最多 `+1`，恢复与最终验收逐条校验 ID 顺序、before/after、
  reserve 守恒、目标 SQ 和最终 ledger；小数、布尔和负数计数 fail closed。
- 持久化每次 grant 的目标、added 与原始跃迁；grant side effect 后、controller
  checkpoint 前的同进程重试会复用原记录，不会把合法 replay 记成零增量扩容。
- 独立 synthesis-only report agent；内部 graph API 不允许隐式回退到研究 agent。
- citation-free caveat 只能逐字复制代码按 plan 状态生成的白名单模板。
- reviewer 必须成功且早于最终写入。
- 每次 CLI 调用独立的 run-scoped 产物目录。

### 03D 明确没有实现

- 不从 low 动态切到 medium/high/xhigh。
- 不在运行中更换主模型或 worker model。
- 不在运行中把 single 切成 multi，或反向切换。
- 不由 controller 自动生成或改写搜索 query。
- 不做大规模质量/成本 benchmark。
- 不做 Claim 与 quote 的自动语义蕴含判断。
- 不把宽问题的每个语义 facet 变成机器可验证的完成项。

把这些边界写清楚很重要：动态模型/topology 会同时改变工具暴露、researcher
门槛、reviewer 要求和 checkpoint 语义。03D 选择先闭合“预算与重试控制”这一条
可独立验证的纵切，不做 prompt-only 的假升级。

## 4. 新的外层调用链

```text
START
  |
  v
plan
  |
  v
select ------------------------------+
  |                                  |
  v                                  |
prepare_research                     |
  |                                  |
  v                                  |
research_agent                       |
  |                                  |
  v                                  |
evaluate                             |
  |                                  |
  +-- strategy=fixed ----------------+
  |
  +-- strategy=adaptive
          |
          v
    adaptive_control
          |
          +-- continue / expand_budget / stop_subquestion --> select
          |
          +-- finish_success / finish_partial / fail_closed --> prepare_report
                                                               |
                                                               v
                                                          report_agent
                                                               |
                                                               v
                                                             finish
```

`adaptive_control` 是普通外层 LangGraph 节点，因此 decision history 会随其他
外层状态一起 checkpoint。它不调用 LLM；同一状态必然产生同一类 action。

## 5. effort 是硬 ceiling

effort 的全局上限保持不变：

| effort | `max_searches` | `max_fetches` | 最少佐证来源组 | 最多 SQ |
| --- | ---: | ---: | ---: | ---: |
| low | 2 | 3 | 2 | 1 |
| medium | 4 | 6 | 2 | 2 |
| high | 8 | 9 | 3 | 4 |
| xhigh | 12 | 14 | 4 | 5 |

fixed 和 adaptive 只改变“何时把上限暴露给 SQ”，不改变上限本身。

### fixed

fixed 仍用 03C 语义：plan 创建后把全部 effort 预算确定性切给各 SQ。它不经过
controller，`max_escalations` 的 effective value 强制为 0。

### adaptive baseline + reserve

adaptive 创建 plan 时先给每个 SQ：

```text
max_searches = 1
max_fetches  = 1
```

剩余部分进入 plan-level reserve：

```text
reserve_searches = effort.max_searches - sum(SQ search grants)
reserve_fetches  = effort.max_fetches  - sum(SQ fetch grants)
```

例如 medium 有两个 SQ 时：

```text
SQ1 baseline: 1 search / 1 fetch
SQ2 baseline: 1 search / 1 fetch
reserve:      2 search / 4 fetch
hard ceiling: 4 search / 6 fetch
```

`grant_subquestion(decision_id, SQ, ...)` 只增加该 SQ 的上限，不重置已经消费的
调用数，也不能使所有 SQ grant 总和超过 effort ceiling。相同 `decision_id`
重放时返回 `idempotent_replay=true`，不会重复扩容。

adaptive 要求 SQ 数不超过可为每个 SQ 保留一组 search/fetch baseline 的数量；
现有 effort policy 的 `max_subquestions` 满足这个约束。

## 6. 结构化 ToolAttempt ledger

每次受预算包装的 `web_search` 或 `fetch_url` 都追加一条有序记录：

```text
attempt_id / sequence / subquestion_id
tool / target
outcome / failure_class / retryable
status / error
provider_outcome / nonempty_search / relevant_search
evidence_producing_search
```

- `target` 对 search 是 query，对 fetch 是 URL。
- 成功、低相关、内容不足、provider/network/http 错误和预算拒绝都会登记。
- `A#` 单调递增；checkpoint 恢复后从已见最大 sequence 继续。
- 新 plan 会清空 plan-local attempts 并从 `A1` 重新开始。
- `provider_successes`、`nonempty_searches`、`relevant_searches` 和
  `evidence_producing_searches` 是四个不同指标。旧字段
  `successful_searches` 仅保留为 `nonempty_searches` 的 deprecated alias。
- 一次搜索即使返回多个相关结果，也只增加一次 `relevant_searches`；非空但
  全部低相关的结果不会帮助 SQ 完成。预算拒绝记为
  `provider_outcome=not_called`，不是 provider failure。

当前分类：

| 情况 | outcome 示例 | failure class | retryable |
| --- | --- | --- | --- |
| 正常成功 | `success` | `none` | false |
| 搜索正常返回但无相关结果 | `low_relevance` | `content` | true |
| SQ 或 plan 无可用额度 | `budget_exceeded` | `budget` | false |
| URL 安全策略拒绝 | `rejected` | `safety` | false |
| 页面正文不足 | `insufficient_content` | `content` | true |
| timeout/connect/request 错误 | `error` | `network` | 由 payload 给出，默认 true |
| 搜索 provider 异常 | `error` | `provider` | 由 payload 给出，默认 true |
| HTTP 状态错误 | `error` | `http` | 由 payload 给出，默认 true |
| 无法识别 | `error` | `unknown` | 由 payload 给出，默认 true |

`retryable` 是结构化提示，不是自动重试承诺。当前 controller 把本周期的
`network/provider` attempt 汇总成 `provider_failure`；实际下一条 query 或备用
URL 仍由 research model 选择。

## 7. controller 的输入、动作与护栏

### assessment 信号

每轮 evaluate 后，代码产生 `ControlAssessment`：

```text
当前 SQ 与 cycle
本轮新增 S#/C#/E#/X#/A#
当前 SQ 剩余 search/fetch grant
plan-level search/fetch reserve
no-progress streak
reason_codes
Evidence Graph integrity errors
```

主要 reason codes 包括：

- `claim_gap`
- `corroborating_source_gap`
- `relevant_search_gap`
- `new_conflict`
- `provider_failure`
- `no_progress`
- `slice_exhausted`
- `hard_ceiling_reached`
- `integrity_failure`

### 决策顺序

controller 的高层优先级是：

1. Evidence Graph 有完整性错误：`fail_closed`。
2. plan 已完成：`finish_success`。
3. 没有可研究 SQ：`finish_partial`。
4. 当前 grant 仍能修复缺口：`continue`。
5. 新登记状态需要一次模型侧更新：允许有界 `continue`。
6. 当前 slice 已耗尽、reserve 能修复缺口、且 escalation 未到上限：
   `expand_budget`。
7. cycle、SQ、escalation 或硬预算 ceiling 到达：`stop_subquestion`。

无新 attempt 的连续无进展也有停止护栏，防止模型只在状态工具之间空转。
research cycle 的总上限为：

```text
max_subquestions * (2 + 2 * effective_max_escalations)
```

### grant 粒度

一次 `expand_budget` 最多给当前 SQ：

- 相关搜索或来源缺口：`+1 search`
- Claim/佐证来源组缺口：同时最多 `+1 fetch`

实际增量还受 reserve 限制。controller 会保存完整的
`budget_before/budget_after`，并核对：

- decision ID 无重复；
- `escalation_count == expand_budget decision 数`；
- escalation 不超过 `--max-escalations`；
- `applied_grant_ids` 数量与 escalation history 一致。

这些是 CLI 验收的一部分，不只是日志。

## 8. checkpoint 与 policy fingerprint

`AdaptiveControlState` 保存：

```text
schema_version / strategy / config_fingerprint
hard_effort / pinned_model / pinned_topology
max_escalations / escalation_count
escalations_by_subquestion
no_progress_streak_by_subquestion
last_assessment / decision_history / stop_reason
```

policy fingerprint 对以下字段做 canonical JSON + SHA-256：

```text
fingerprint schema
strategy
effort
requested mode
resolved topology
main model
worker model
effective max_escalations
```

如果 checkpoint 仍有 pending node，CLI 会先比较 fingerprint。任一字段变化都会
拒绝恢复，并提示使用原始 strategy、effort、mode、模型和
`max-escalations`。budget restore 同时校验：

- 保存的 effort/strategy 与当前策略一致；
- SQ grant 总和不超过硬上限；
- SQ usage 不超过已授予额度。

controller 节点前中断并恢复时，外层会继续同一个 pending
`adaptive_control`；确定性 decision ID 与 `applied_grant_ids` 保证同一 grant
只应用一次。旧 pending checkpoint 若没有 controller state，只能用
`strategy=fixed` 恢复。

## 9. 控制事件与审计产物

### events.jsonl

03D 增加以下事件：

| event | 含义 |
| --- | --- |
| `control_assessed` | 保存 decision ID、action、reason codes 与完整 assessment |
| `budget_expanded` | reserve 已释放，保存 grant 与 budget before/after |
| `adaptive_continued` | 不扩容，继续下一次选择/研究 |
| `adaptive_stopped` | 停止 SQ 或结束成功/partial |
| `control_integrity_failed` | Evidence Graph 错误触发 fail-closed |

### control.json

每个 run 都写出完整 `adaptive_control` checkpoint snapshot。fixed 也会生成该
文件，但 strategy 为 fixed、effective max escalation 为 0，evaluate 不经过
adaptive controller。

### run.json

03D 在顶层或 `adaptive_control` 中记录：

```text
output_dir
strategy / effort / max_escalations
pinned model / topology
escalation_count / stop_reason
decisions / last_assessment / control_path
```

完整 budget snapshot 仍位于 `run.json.budget`，其中包括 baseline/grant/reserve、
`applied_grant_ids` 和 `tool_attempts`。

### run-scoped output

每次 CLI 调用创建：

```text
output/runs/<sanitized-thread-prefix>-<thread-hash>/<UTC-run-id>/
```

该目录同时是 Deep Agents 的虚拟文件 backend root 和最终审计目录。相同 thread
的多次调用不会覆盖彼此；不同原始 thread 即使清洗后前缀相同，也由 hash 隔离。
`../`、斜杠和非 ASCII thread 字符不能逃离 `output/runs/`。共享的
`output/checkpoints.sqlite` 不在 run 目录内，因为它需要跨运行恢复 thread。

每个 run 内包含：

```text
report.md
trace.json
sources.json
evidence.json
plan.json
events.jsonl
control.json
run.json
```

## 10. reviewer success-before-write 修补

high/xhigh multi 仍要求 reviewer，但 03D 把验收条件收紧为一条完整 trace：

```text
report phase task(subagent_type="reviewer")
  -> 相同 tool-call ID、内容非空的 ToolMessage(status="success")
  -> 之后才出现最终 write_file
```

以下情况都不满足门槛：

- 只有 reviewer tool call，没有结果；
- reviewer ToolMessage 是 error；
- reviewer 成功回执内容为空；
- 回执的 tool-call ID 不匹配；
- reviewer 在最终 `write_file` 之后才成功。

trace 现在保留每个 tool result 的 `status` 和 `phase`，因此该门槛可以独立审计。
这仍不表示代码理解了 reviewer 建议的语义质量；它只修复成功性和时序闭包。

## 11. 代码索引

- `adaptive_control.py`：assessment、纯函数 action 决策、幂等 decision history。
- `agent_policy.py`：effort 硬策略、静态 topology 路由、policy fingerprint。
- `research_state.py`：`ToolAttempt`、`ControlAssessment/Decision` 和
  `AdaptiveControlState` 类型。
- `research_graph.py`：`evaluate -> adaptive_control` 节点、事件、恢复和路由。
- `evidence_graph.py`：严格 Claim/Source 行映射与状态派生 caveat allowlist。
- `search_agent.py`：baseline/reserve、attempt 分类、grant、CLI 参数、恢复验收、
  decision/ledger 审计、reviewer 时序门槛和 run-scoped artifacts。
- `tests/test_adaptive_control.py`：控制器纯函数、硬上限、幂等 grant、策略漂移。
- `tests/test_adaptive_graph.py`：真实外层图的缺口扩容和 controller 前恢复。
- `tests/test_stage03d_safety.py`：reviewer 时序、trace status/phase、run 隔离、
  grant 跃迁、空 history 合法恢复、plan/budget scope 与 caveat 隔离。

## 12. 离线验证

完整命令：

```bash
.venv/bin/python -m unittest discover -s tests -v

ruff check \
  adaptive_control.py agent_policy.py research_state.py research_graph.py \
  evidence_graph.py telemetry.py search_agent.py tests

ruff format --check \
  adaptive_control.py agent_policy.py research_state.py research_graph.py \
  evidence_graph.py telemetry.py search_agent.py tests
```

当前验证结果：

```text
Ran 144 tests
OK
```

144 项包括此前 03C/03D 回归，并新增或强化：

- fixed/adaptive baseline 语义；
- reserve 的单调、幂等和 effort 硬上限；
- success、低相关、内容、network/provider/http、安全和预算分类；
- provider/nonempty/relevant/evidence-producing 的原子计数与后端自报防污染；
- attempt ledger 与 checkpoint counters 的一致性及旧指标 unavailable；
- hostname/content revision/支持性佐证组的保守分组；
- 字符/字节/gzip 截断、Content-Length 与 captured hash scope；
- ledger 复核的 structural closure、coverage alias 与 legacy source-only null 语义；
- 完整性错误优先 fail-closed；
- Claim/来源/搜索缺口驱动的 continue/expand/stop；
- escalation/cycle/no-progress ceiling；
- controller decision ID、budget before/after 和 history 一致性；
- 单次 grant 的每维 `+1` 上限、严格顺序、连续跃迁和最终 ledger 对齐；
- pending controller 恢复只应用一次 grant，首个 decision 前中断也可安全恢复；
- grant side effect 与 controller checkpoint 之间异常后的原跃迁幂等 replay；
- effort/strategy/grant/usage、非整数计数与 plan/budget SQ scope 漂移拒绝；
- 独立无研究工具的 synthesis agent 与 citation-free caveat 白名单；
- reviewer 成功且早于最终写入；
- trace tool-result status/phase；
- thread/run 输出隔离与路径清洗。

离线测试验证确定性机制；真实 provider、页面、模型、恢复和 CLI 路径由下一节
的 release smoke 单独覆盖。

## 13. 真实联网 smoke

最终 release smoke 使用一个原子 SQ 和两个 Python 官方页面，命令为：

```bash
.venv/bin/python search_agent.py \
  --strategy adaptive \
  --max-escalations 2 \
  --effort low \
  --mode single \
  --model gpt-5.4-nano \
  --worker-model deepseek-v4-flash \
  --thread-id stage03d-release-final-20260717-a1 \
  --no-stream \
  --print-report \
  "这是 Stage 03D 最终窄范围机制验收。只建立一个原子子问题：核验 Python 3.13 的 free-threaded CPython（禁用 GIL）仍属于实验性功能。必须至少搜索一次，并分别抓取和登记以下两个 Python 官方页面中的 exact quote：https://www.python.org/downloads/release/python-3130/ 与 https://docs.python.org/3/whatsnew/3.13.html；最终只报告 canonical Claim，不得添加任何自由发挥的 caveat，也不得研究其他 Python 版本。若 baseline 不足，让外层 adaptive controller 释放 reserve。"
```

产物目录：

```text
output/runs/stage03d-release-final-20260717-a1-12314473/
  20260717T051353.769171Z-e343e56c/
```

验收结果：

- `validation.status=passed`，`structural_subquestion_coverage=1.0`
  （旧 `coverage` 为 deprecated alias）。
- `strategy=adaptive`、`effort=low`、`topology=single`。
- `search_calls=1/2`、历史字段 `successful_searches=1`（现明确为 nonempty
  alias）、`fetch_calls=2/3`。
- 2 个不同 hostname 且正文不重复的支持来源组、2 个 supported Claim、
  2 个 Evidence、0 个完整性错误。
- decision 序列为 `continue -> expand_budget -> finish_success`；
  `escalation_count=1`，grant ID 与逐步预算跃迁一致。
- 最终报告只有四个 H2，`Conflicts and Caveats` 为空，没有自由 caveat；当前代码
  对该产物重跑 adaptive audit、plan/budget scope audit 和 report mapping 均为零错误。
- 首次 sandbox 联网受限时 checkpoint 停在首个 controller decision 之前；解除
  联网限制后同一 thread 从空 decision history 合法恢复并完成，额外验证了恢复门槛。

此前的 medium/adaptive 两 SQ smoke 也曾以 `escalation_count=0`、
`stop_reason=plan_completed` 通过，证明 baseline 足够时不会为“看起来更努力”而
强制扩容。该联网结果是 03D 当时的历史 smoke，不是正式 benchmark；本轮
benchmark-readiness 结论以最新离线回归和重新生成的标准评测记录为准。

## 14. 已知限制

### 14.1 crash-mid-inner 不是 exactly once

checkpoint 仍只属于外层图；内层 Deep Agent 使用 `checkpointer=False`。只有
`research_agent` 返回后，外层才把消息、budget、Evidence 和 attempts 作为一个
节点结果提交。

如果进程在内层模型/工具循环中崩溃：

- 当前外层节点尚未 commit；
- 恢复后可能重跑整个 inner node；
- 网络搜索或抓取可能再次发生；
- 进程内 page cache 丢失；
- controller grant ID 的幂等不能扩展成所有外部工具的 exactly-once 语义。

因此，03D 已验证的是“controller 节点前恢复不重复 grant”，不是任意崩溃点的
端到端无重复副作用。

### 14.2 structural subquestion coverage 没有 semantic facets

`structural_subquestion_coverage` 按 SQ 是否形成合法
Claim-Evidence-Source 闭包计算；只有当前 ledger 重新核验过的
`structural_closure_validated=true` SQ 才计数，伪造 C#/S# 或缺少逐 SQ
相关搜索都不能产生 coverage，checkpoint 恢复时也会重新审计。旧 `coverage`
仅是 deprecated alias。代码目前没有把
“核心团队、论文、项目、开源、合作、目标、参与方式、公开进展”等宽问题字段
展开成必须逐项满足的机器可读 facet。

结果是：

- 一个过宽 SQ 可能只登记其中一个窄 Claim；
- 来源数和 Claim 闭包可以满足，但宽 SQ 的语义字段仍不完整；
- controller 会看到 `claim_gap/source_gap/search_gap`，却看不到未建模的 facet；
- `supports/contradicts` 仍由模型标注，exact quote 不等于语义蕴含证明。

当前缓解方式是把真实验收提示拆成原子 SQ。该指标不能代表答案准确率、语义
覆盖率、完整性或 citation entailment；宽问题即使达到 1.0 仍需外部 benchmark
或 judge。旧 source-only schema 无法证明 Claim 闭包时输出 `null/unavailable`。
后续可加入显式 facet schema、Claim-to-facet 映射和独立 semantic reviewer。

### 14.3 其他边界

- controller 不自动改写 query；attempt 分类只为下一轮提供可审计信号。
- `retryable=true` 不保证同一目标值得重试。
- 动态 model/topology 仍未实现，并被 fingerprint 明确钉住。
- reviewer success-before-write 只证明审稿调用成功且时序正确，不证明建议被
  充分采纳。
- token 统计仍可能漏 planner 和部分 nested subagent，用量不等于账单。
- 页面抓取继续受 JavaScript、PDF、字符截断和 HTML 提取边界影响。

## 15. 下一阶段假设

03D 后更合理的后续顺序是：

1. 为宽问题增加显式 semantic facets 与 Claim-to-facet coverage。
2. 增加 Claim/quote 语义支持 reviewer 和近义 Claim/冲突归一化。
3. 改善 inner-run durability 或为有副作用的工具提供更强 idempotency。
4. 在机制闭合后，再做 fixed/adaptive 的质量、成本、成功率和恢复 benchmark。

大规模 benchmark 不属于 03D 的完成条件；真实 smoke 只用于验证当前纵切能通过
真实 provider、模型、工具、checkpoint 和 CLI 验收路径。
