# TongAgent 评测指标语义

本文定义统一 evaluation harness 的字段和 TongAgent 原生研究指标。共同原则是：

- 观察不到的值写 JSON `null`，不能用 0 代替；
- 0 表示“已知且观察到零次/零值”；
- 内部流程指标不能改名包装成答案质量；
- fixture 结果只能解释为 smoke；
- B1/B2 没有 TongAgent Evidence Graph，因此 `evidence_count` 和
  `structural_subquestion_coverage` 必须为 `null`。

## RunResult 完整字段

### 身份、配置与时间

| 名称 | 定义与计算 | 使用范围 | 不能代表什么 |
| --- | --- | --- | --- |
| `run_id` | worker 为一个 attempt 分配的唯一运行 ID | 定位 trace、metrics、failure | 不表示实验配置相同，也不是 task ID |
| `task_id` | JSONL 中经过安全校验的 `id` | 连接输入与结果 | 不表示问题文本或数据 release 未变化 |
| `system_id` | `simple_react`、`vanilla_deepagents` 或 `tongagent` | 分组 B1/B2/B3 | 不等于版本；必须结合 git/config |
| `git_sha` | 调度时本地 checkout 的 `HEAD` | 代码 provenance | 不证明工作树干净，也不包含未提交 diff |
| `resolved_config` | 本 attempt 的完整、非秘密配置；显式保留 `null` | 重放与审计 | 不包含凭证值，也不证明外部服务状态相同 |
| `config_fingerprint` | canonical JSON 的 SHA-256；覆盖共同配置、system ID 和 system options，排除 artifact 路径及 fingerprint 自身 | 判断同一系统配置能否安全 resume | 不表示不同系统公平；公平比较看下一字段 |
| `fairness_fingerprint` | 排除 `system_id`、`system_options` 和 artifact 路径后的配置 SHA-256 | 拒绝混合模型、工具、数据、backend、judge、seed 或预算不同的系统 | 不证明 prompt/框架控制流相同；它们是被比较的处理变量 |
| `started_at` / `finished_at` | UTC wall-clock 时间戳 | 排查运行时间与外部事件 | 不用于执行 deadline；deadline 使用单调时钟 |
| `wall_time_seconds` | attempt 内单调计时的非负耗时；父进程超时时由调度层记录 | 资源与延迟对比 | 不是纯模型 latency；包含框架、工具、IO，且受机器/网络负载影响 |
| `artifact_directory` | 当前 attempt 的绝对输出目录 | 定位完整证据 | 不进入 fingerprint，不应用于跨机器实验身份 |

`git_sha` 不能替代 `git status`。若正式实验从含未提交修改的工作树运行，必须另行
保存 diff 或先形成可引用 commit。

### 答案、引用与工具

| 名称 | 定义与计算 | 使用范围 | 不能代表什么 |
| --- | --- | --- | --- |
| `final_answer` | adapter 提取的 terminal 模型答案；没有合法终态时为 `null` | 后续 scorer、人工审阅 | 非空不等于正确或完整 |
| `citations` | 最终答案中可映射到成功抓取的 citation 记录；B3 进一步来自报告中的 Claim–Evidence–Source 边 | provenance 审计和引用数 | citation 数不等于引用正确、蕴含或来源权威 |
| `tool_calls` | 按调用次序保存的脱敏 model-facing 工具记录，含参数、时序、status、摘要 result/failure | 调用审计、失败定位 | 长度不等于成功调用数；预算拒绝和非网络工具也可能在内 |
| `search_calls` | 共同执行预算实际接纳的 search 调用数 | 三系统成本/行为对比 | 不等于 provider success、非空、相关或产证搜索；预算拒绝且未获 reservation 的请求不增加它 |
| `fetch_calls` | 共同执行预算实际接纳的 fetch/open 调用数 | 三系统成本/行为对比 | 不等于抓取成功、合格证据或独立来源数 |
| `relevant_searches` | canonical search attempt 中至少一个结果达到当前代码相关性门槛的次数；每次搜索最多 +1 | 三系统共享的搜索诊断；B3 完成/控制的必要信号之一 | 词法相关性不证明语义相关、事实正确或最终产证 |

`ToolCall.status` 的取值为：

- `success`：工具返回且没有被 canonical semantic failure 判定为错误；
- `error`：执行异常或结果状态被判定为错误，并带 `failure`；
- `budget_exceeded`：共同预算未接纳调用，provider 未被调用；
- `not_called`：schema 保留的未调用状态；不能算成功。

canonical trace 对正文和其他大文本设有上限；超限值保存字符数与 SHA-256 摘要。
因此 trace 适合审计调用与状态，不是网页正文归档。

### Evidence、答案评分与资源

| 名称 | 定义与计算 | 使用范围 | 不能代表什么 |
| --- | --- | --- | --- |
| `evidence_count` | B3 ledger 中 Evidence unit 记录数；图完整性由独立 validator 决定，失败记录不会靠这个 count 变合法；B1/B2 为 `null` | B3 内部图规模与回归 | 不能跨系统当质量分；更多 evidence 不必然更准确 |
| `structural_subquestion_coverage` | B3 中满足结构闭包的 SQ 数 / 全部 SQ 数，四舍五入到 4 位；无当前 schema 时为 `null` | B3 计划/证据闭包回归 | 不代表答案准确率、语义 facet 覆盖、完整性或 citation entailment |
| `token_usage` | provider usage metadata 的聚合对象；完全不可用时整个对象为 `null` | 资源观察和预算核算 | 不是成本；部分字段缺失时不能补 0 |
| `estimated_cost` | 已知定价协议下估算的非负费用；当前没有可信 price registry 时为 `null` | 将来成本对比 | `null` 不表示免费；也不能从 token 数直接猜价格 |
| `normalized_exact_match` | 有参考答案时，对 prediction/reference 分别做 NFKC、casefold、空白折叠后进行 whole-string 等值；无参考答案为 `null`，有参考但无答案为 `false` | 短、规范答案的保守诊断 | 不适合长报告；不接受 substring，不代表语义等价 |
| `judge_score` | 预注册、版本固定的外部 judge 输出，范围 `[0,1]`；没有 judge 为 `null` | 需要 rubric/语义评估的 benchmark | 不能由 B3 内部 coverage 或 evidence 数回填；单一分数也不能省略 judge 原始记录 |
| `judge_result` | judge 的规范化结果，含与 `judge_score` 相同的 `score`、可空 `rationale` 和 JSON metadata；没有 judge 时为 `null` | 保存可审阅的评分上下文 | 仍不等于完整 provider 原始响应；实现应在 metadata 中保留足够 provenance |
| `fixture_smoke` | backend 为 fixture 时为 `true`，live 时为 `false` | 阻止把离线回归冒充正式结果 | `false` 只说明不是 fixture，不自动证明实验正式、公平或规模充分 |

`judge_score` 非空时 `judge_result` 必须同时存在且两个 score 完全一致。当前 CLI
没有内置外部 judge，因此默认二者均为 `null`；配置一个未注册的
`judge.id/version` 会在运行前 fail fast，不会退回内部结构分。

`TokenUsage` 子字段：

| 字段 | 含义 | 空值规则 |
| --- | --- | --- |
| `input_tokens` | provider 报告的输入 token 聚合 | provider 未报告则 `null` |
| `output_tokens` | provider 报告的输出 token 聚合 | provider 未报告则 `null` |
| `total_tokens` | provider 报告的总 token 聚合 | 未报告则 `null`；若 input/output 都已知，不能小于两者之和 |
| `cached_input_tokens` | provider 报告的 cache-read 输入 token | 未报告则 `null`，不能把无 cache 当成 0 |
| `reasoning_tokens` | provider 报告的 reasoning token | 未报告则 `null` |

fixture 模型生成的 usage 是确定性的近似 token 计数，用于测试 accounting 和预算
路径；它不是任何线上 provider 的 tokenizer 或账单。live 模式只应解释实际
provider 返回的字段。

模型调用在 provider I/O 前以“估算输入 + `max_output_tokens`”创建 reservation：

- provider 返回可核算 usage 后，释放 reservation 并按 actual token 结算；
- provider 成功但 usage 不可核算时，预算保守收取整笔 reservation，而
  `RunResult.token_usage` 仍为 `null` 或 partial；
- provider 调用抛错且无已知 usage 时取消 reservation；
- actual 超过 reservation 时保留 overrun，若越过 ceiling 则把 run 标为
  token budget exhausted，不能丢弃已经发生的消费。

`native/budget.json` 因此区分 provider-reported `total_tokens`、未知 usage 的
`estimated_token_charges`、两者合计 `accounted_tokens`、尚未结算的
`reserved_tokens` 和 `token_budget_exhausted`。这些预算字段不等于账单，也
不能回填 `RunResult.token_usage`。

### 完成与失败

| 名称 | 定义与计算 | 使用范围 | 不能代表什么 |
| --- | --- | --- | --- |
| `completion_status` | attempt 的 terminal 状态 | 成功率、失败分布、resume | `completed` 只表示系统协议完成，不表示答案正确 |
| `failure_type` | terminal failure 的稳定类别；成功或 honest partial 可为 `null` | 聚合失败分布 | 不汇总所有已处理工具失败；后者查 tool/native trace |
| `failure` | failure type、脱敏 message、stage、retryable 和 bounded details | 定位根因 | `retryable=true` 不是自动重试承诺 |

`completion_status`：

- `completed`：有最终答案，且没有 terminal failure；B3 还通过其原生完整性与
  完成验收。
- `partial`：产生可用输出但 B3 的研究计划或完成条件没有全部满足；这是诚实的
  不完整状态，不伪造 failure。
- `failed`：runner、模型、工具、fixture 或输出协议失败。
- `budget_exhausted`：共同 model/token/tool 预算被拒绝。
- `timed_out`：共同单调 deadline 或父进程 deadline 到达。
- `interrupted`：明确记录的中断终态；没有 `result.json` 的崩溃目录只是未完成
  attempt，不应伪造为这一状态。

`failure_type` 当前稳定取值：

`budget_exhausted`、`deadline_exceeded`、`fetch_error`、
`fixture_not_found`、`interrupted`、`invalid_output`、`invalid_task`、
`judge_error`、`model_error`、`runner_error`、`search_error`、`tool_error`。

## 搜索语义阶梯

搜索指标按一次 canonical `web_search` attempt 计算，而不是按返回结果条数：

```text
search_calls
  └─ provider_successes
       └─ nonempty_searches
            ├─ relevant_searches
            └─ evidence_producing_searches（需要后续 Evidence）
```

`relevant_searches` 由搜索时的轻量相关性分数决定；
`evidence_producing_searches` 由后续是否登记有效 Evidence 决定，二者不是彼此
的子集。一个词法分数未过门槛的结果仍可能被后续严格 quote 校验产证；反之，
相关搜索也可能没有产证。旧 checkpoint 的不可用字段必须保留 `null`，不能为了
满足某种关系而推断历史值。

| 名称 | 定义 | 计算方法与范围 | 不能代表什么 |
| --- | --- | --- | --- |
| `provider_successes` | provider 被实际调用，canonical 结果状态为成功 | 每个 search attempt 最多 +1；预算拒绝/安全拒绝为 `provider_outcome=not_called`，网络/provider 异常为 failure | 不保证有结果、相关或可抓取 |
| `nonempty_searches` | provider success 且至少一个结果含 URL | 每个 search attempt 最多 +1 | 非空结果可能全部无关 |
| `successful_searches` | deprecated compatibility alias | 必须与 `nonempty_searches` 同义；新完成逻辑不得使用它冒充 relevant | 名称中的 successful 不表示有效研究成功 |
| `relevant_searches` | 至少一个带 URL 的结果的代码计算 `relevance_score` 达到门槛 | canonical 代码从逐结果分数计算；一次有多个相关结果仍只 +1；后端自报值不能覆盖 canonical 值 | 轻量词法门槛不是语义 judge |
| `evidence_producing_searches` | 该搜索发现的 URL 后来产生新的、通过校验的 Evidence | Evidence 登记时回溯匹配 search attempt，首次置位并 +1；同一搜索产生多个 Evidence 仍只 +1；resume/replay 保持幂等 | 不表示 Evidence 支持的 claim 为真 |
| provider/network failure | provider 被调用但异常或返回错误状态 | 保存在有序 attempt ledger 的 `outcome`、`failure_class`、`retryable`、status/error 中；不是“成功数的负数” | 单次失败不自动等于整个 run 失败 |

无结果和非空但低相关都可能是 provider success；后者会增加
`nonempty_searches`，但绝不能增加 `relevant_searches`，不能帮助 SQ 达到
covered，也不能让 adaptive controller 因“搜索成功”提前停止。

每条 B3 search attempt 还保存：

- `attempt_id` / 单调 `sequence`；
- 当前 `subquestion_id` 与 query；
- `provider_outcome=not_called|success|failure`；
- `result_urls`、`relevant_result_urls` 和结果计数；
- `evidence_producing_search`；
- canonical 值与 provider 自报值不一致时的 `semantic_mismatches`；
- `failure_class=none|budget|safety|content|network|provider|http|unknown`。

这些字段使 retry 与 checkpoint/resume 可由 ledger 重算，避免重复计数。

## 来源 revision、host 与佐证组

三个概念必须分开：

### `distinct_content_revisions`

它是候选 supporting Evidence 实际绑定的
`source_content_sha256` 去重集合；count 是集合大小。hash 对应本轮**捕获并规范化
后的可见文本**，不是天然的完整网页版本。

每次成功 fetch 的 source revision 另外记录：

- `captured_content_sha256` / 兼容字段 `content_sha256`；
- `content_sha256_complete`；
- `fetched_at`；
- 返回字符数、观察到的正文长度、下载字节数、HTTP Content-Length/Encoding；
- `truncated` 和 `truncation_reasons`。

当下载字节或返回字符被截断时，hash 只能称为 captured-content hash；
`content_sha256_complete=false`。相同 URL 内容变化会追加 revision，不覆盖早期
Evidence 绑定的 hash。

不能代表：

- 不同 revision 是不同出版者或独立来源；
- 相同 hash 的页面在所有时间都相同；
- hash 证明发布者身份、真实性、时效性或 claim 蕴含。

### `distinct_source_hosts`

这是 participating supporting Evidence 来源的规范化 hostname 集合。当前轻量
规则：

- hostname casefold、去末尾点并转 IDNA；
- 只移除前缀 `www.`；
- 合法 IP 使用规范化文本；
- 其他子域保持不同；
- 不使用 `host.endswith(domain)`，因此 `evilbigai.ai` 不会匹配
  `bigai.ai`；
- 不能安全解析的 hostname 进入 `unavailable_source_ids`，不增加 count。

不能代表：

- 注册域、publisher 或法律主体；
- 不同 host 的编辑独立性；
- 同一机构的不同子域已被正确合并。

### `corroborating_source_groups`

只从满足以下条件的 Evidence 构组：

- stance 为 `supports`；
- source/claim 位于当前候选范围；
- Evidence 绑定合法 revision hash；
- source 有可安全规范化的 hostname。

若两个来源 hostname 相同，或其 Evidence revision hash 有完全相同内容，它们
归入同一组；该关系取传递闭包。每组只选一个
`representative_source_id` 计入完成门槛。同 host 的不同页面不会自动算两份
佐证；不同 host 的精确镜像也不会算两份；`contradicts` Evidence 不能补足支持
来源门槛。

不能代表：

- group 之间已被证明具有不同 publisher、作者或信息源；
- supporting stance 经过语义 entailment judge；
- 多组一致就证明 claim 为真。

当前名称是“保守佐证来源组”，不是“独立来源”。旧
`independent_evidence_source_ids` 只作为兼容 alias，不能恢复旧的 revision-only
解释。

## `structural_subquestion_coverage`

定义：

```text
通过结构闭包审计的 covered SQ 数 / 当前 plan 的全部 SQ 数
```

当前 evidence schema 下，一个 SQ 计入分子至少需要：

- 状态为 `covered`；
- `structural_closure_validated=true`；
- 有 canonical `claim_ids` 与 `evidence_source_ids`；
- ledger 审计确认 Claim 属于该 SQ，存在 supporting Evidence，Claim–Evidence–
  Source 边一致；
- SQ 自己的预算 scope 中有相关搜索，不能借其他 SQ 的 surplus 串账。

checkpoint 恢复会重新审计，不能只信持久化的 true。当前 schema 且无 SQ 时值为
0.0；旧 schema 无法证明这些语义时为 `null`。旧 `coverage` 字段只是 deprecated
alias，必须与本指标相同并明确 structural。

使用范围：

- B3 计划/证据闭包回归；
- adaptive controller 的结构化缺口信号；
- 检查无关搜索、跨 SQ 串账和缺 Evidence 是否导致虚假完成。

不能代表：

- 最终答案准确率；
- 用户问题的语义 facet coverage；
- 宽问题的内容完整性；
- citation entailment、来源质量或冲突裁决正确性。

例如一个 SQ 同时要求“背景、团队、项目、进展”，只要有一个合格 Claim 闭包就
可能满足结构条件；在没有外部 facet annotations/judge 时，代码不能声称四项
语义内容均已覆盖。

## 聚合指标

`summary.json` 的系统级聚合包括：

- `runs`；
- `completion_distribution` 与 terminal `failure_distribution`；
- exact match 可用数、match 数和可用样本上的 rate；
- tool/search/fetch 总数与 mean tool calls；
- wall time 总数与 mean；
- token/cost 的已知 run 数。

token 或 cost 只有在该系统**每个选中结果**都有已知值时才计算总量，否则总量为
`null`，同时保留 `known_*_runs`。这避免用已知子集的和伪装成完整成本。

聚合器选择每个 task/system 的最新合法 terminal result，并单列没有
`result.json` 的 incomplete attempt。它拒绝损坏结果和 mixed fairness
fingerprint。聚合统计仍不能修复样本偏差、失败选择偏差或 judge 偏差；正式报告
必须同时展示失败分布和空值范围。

## 解读清单

发布或比较结果前至少确认：

1. 所有行具有同一 `fairness_fingerprint`。
2. fixture 结果的 `fixture_smoke=true`，标题明确写 smoke。
3. B1/B2 的 Evidence/coverage 为 `null`，没有用 0 惩罚它们。
4. `relevant_searches` 没有被 `nonempty_searches` 或 deprecated
   `successful_searches` 替代。
5. token/cost 的 `null` 没有补 0。
6. exact match 只用于适合 whole-string 的参考答案。
7. 长报告准确性、完整性和 citation entailment 来自固定版本外部 judge/官方
   evaluator，而不是内部结构指标。
8. 报告同时包含 partial、failed、budget-exhausted、timed-out 和 incomplete
   attempt，不只展示成功样本。
