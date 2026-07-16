# Stage 02：分级、多 Agent、可恢复研究闭环

日期：2026-07-16

状态：已完成，可复现

前一阶段：`stage-01` 真实单 Agent 搜索闭环

## 一句话成果

TongAgent 从“一个能搜索和写报告的 Agent”演进为一个有结构化证据、硬资源预算、可选 single/multi 拓扑、独立 reviewer、SQLite checkpoint 和自动验收的研究 harness。

## 1. 本阶段回答的问题

Stage 01 证明真实 Agent Loop 可以跑通；Stage 02 进一步回答：

- 如何确认引用的是成功读取的页面，而不是搜索摘要或失败页面？
- 简单问题是否必须承担多 Agent 的计算成本？
- 复杂问题如何显式加入 researcher 和 reviewer？
- “低、中、高、极高”如何落实为可审计的资源差异？
- 进程结束后如何恢复同一研究线程？
- 恢复历史时，如何避免旧工具调用污染本次运行验收？

## 2. 已完成能力

| 能力 | 代码行为 | 验证证据 |
| --- | --- | --- |
| 结构化抓取 | `fetch_url` 返回 status、URL、title、content、字符数 | 抓取与失败回归测试 |
| 来源账本 | 成功页面获得稳定 `[S1]` ID，fragment 去重 | `sources.json` 与去重测试 |
| 证据门槛 | 少于 500 字符不算成功来源 | 短页面测试 |
| 硬资源预算 | 搜索/抓取额度由共享锁保护，所有 Agent 共用 | 预算耗尽测试 |
| 可选拓扑 | `--mode single|multi|auto` | 真实 single/multi 轨迹 |
| 显式角色 | researcher 搜证；high/xhigh 的 reviewer 无工具审稿 | `task` 委派轨迹 |
| 四档强度 | `--effort low|medium|high|xhigh` | 策略测试与 `run.json` |
| 持久化状态 | SQLite saver + `thread_id` | 关闭并重开数据库的恢复测试 |
| 单次审计隔离 | 恢复历史供模型使用，但 trace 只记录新消息 | checkpoint 回归测试 |
| 质量验收 | 校验来源数、引用、角色委派、Sources 和真实写文件 | 成功与门槛失败实测 |

## 3. 架构

```text
                    +-----------------------------+
用户问题 ----------> policy(mode, effort, topic) |
                    +--------------+--------------+
                                   |
                  +----------------+----------------+
                  |                                 |
             single topology                   multi topology
                  |                                 |
        主 Agent 直接持有网络工具          主 Agent 不持有网络工具
                  |                                 |
                  |                       task(researcher)
                  |                                 |
                  +----------> 共享 ResearchBudget <+
                               web_search/fetch_url
                                      |
                              [S1] [S2] 证据账本
                                      |
                        high/xhigh: task(reviewer)
                                      |
                                  write_file
                                      |
                 report.md / trace.json / sources.json / run.json
                                      |
                          SQLite checkpoint(thread_id)
```

关键边界：

- single 模式没有 `task`，因此不会暗中变成多 Agent。
- multi 主 Agent 没有网络工具，必须把检索委派给 researcher。
- reviewer 没有工具，只检查主 Agent 传入的草稿与证据。
- researcher 与主流程共享同一个代码级预算，prompt 不能绕过额度。
- 文件后端仍然是真实落盘，但根目录被限制在 `output/`。

## 4. 分级计算策略

| effort | 默认主模型 | 搜索 | 抓取 | 最少成功来源 | 页面字符上限 | reviewer |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| low | `gpt-5.4-nano` | 2 | 3 | 2 | 8,000 | 否 |
| medium | `gpt-5.4-nano` | 4 | 6 | 2 | 12,000 | 否 |
| high | `gpt-5.4-mini` | 8 | 9 | 3 | 15,000 | multi 时必须 |
| xhigh | `gpt-5.4-mini` | 12 | 14 | 4 | 20,000 | multi 时必须 |

这四档同时改变模型、工具预算、上下文规模、输出上限、证据门槛和角色拓扑，比只设置一个“思考强度”字符串更可控。但它目前是应用层计算策略，不等同于底层模型厂商的 reasoning effort 参数。

`auto` 的当前规则是：high/xhigh 固定选择 multi；low/medium 遇到复杂度关键词或较长问题时选择 multi，否则 single。该决策可复现，但只发生在开始运行时。

## 5. checkpoint 语义

默认使用 `output/checkpoints.sqlite`：

- 不传 `--thread-id`：生成新 UUID，默认不会把无关任务混到一起。
- 传相同 `--thread-id`：恢复消息历史，允许跨进程追问。
- `--follow-up`：在一个进程中增加多个连续 turn。
- `--checkpoint-db`：允许选择其他 SQLite 文件。

恢复时先读取旧消息 ID。模型收到完整历史，但 `trace.json`、角色验收和本次工具链只使用新消息。这解决了“有记忆”和“本轮证据可独立审计”之间的冲突。

## 6. 模型兼容性与成本结论

免费模型的真实探测结果：

- `qwen3.6-35b-a3b`：网关没有为该模型启用自动 tool parser，带工具请求返回 HTTP 400，不能驱动当前 Agent Loop。
- `deepseek-v4-flash`：简单工具往返能成功，但进入完整 Deep Agents harness 后会在 ToolMessage 后产生不稳定输出，因此不作为主工具模型。
- 免费 DeepSeek 可以稳定承担无工具 reviewer；low/medium 主模型用 nano，high/xhigh 主模型用 mini。

API 用量通过私有网关的 `/usage` 端点只读查询。精确余额和账户信息不写入公开仓库；后续代码回归默认运行离线单元测试，不消耗 API token，只有明确需要端到端验证时才调用付费主模型。

## 7. 真实路径验证

已经实际观察到三条主路径：

```text
low / single / gpt-5.4-nano
  -> web_search / fetch_url / write_file

medium / multi / gpt-5.4-nano
  -> task(researcher) -> write_file

high / multi / gpt-5.4-mini + free deepseek reviewer
  -> task(researcher) -> task(reviewer) -> write_file
```

high 实验中，reviewer 最终只保留 2 个强来源，而 high 门槛要求 3 个。报告成功写出，但程序将本轮标记为验收失败。这不是崩溃，而是质量门槛真正阻止了“看起来完成、证据却不达标”的结果。

## 8. 离线验证

```text
test_auto_routes_by_effort_and_complexity ... ok
test_explicit_mode_wins ... ok
test_policy_prompt_exposes_enforced_budget ... ok
test_subagent_roles_follow_topology_and_effort ... ok
test_same_thread_recovers_history_but_trace_keeps_current_turn_only ... ok
test_http_403_becomes_a_tool_result ... ok
test_search_budget_is_enforced ... ok
test_short_pages_do_not_count_as_evidence ... ok
test_successful_fetches_receive_stable_source_ids ... ok
test_stream_hides_tool_content_and_keeps_events ... ok

Ran 10 tests ... OK
```

## 9. 创新点判断

“支持多 Agent”或“提供四档强度”本身已经是常见产品能力，单独拿出来不构成很强的技术创新。

更值得发展的创新主线是证据驱动的自适应计算：系统先以低成本配置执行，再根据成功来源不足、来源冲突、覆盖度和 reviewer 反馈，决定是否扩大搜索预算、升级模型或从 single 切换到 multi；最后用质量、费用和延迟实验说明它优于固定配置。

Stage 02 已经提供这条主线需要的四个基础件：

1. 可选择的计算档位。
2. 可切换的 Agent 拓扑。
3. 机器可读的证据与失败状态。
4. 明确的质量验收信号。

尚未实现的是“运行中闭环升级控制器”和系统化对照实验，因此 PPT 中应把 Stage 02 表述为创新基础设施，而不是已经证明创新成立。

## 10. 已知限制与 Stage 03

- auto 只做运行前路由，不会因中途证据不足自动升级。
- 工具预算按当前进程创建，checkpoint 只恢复 LangGraph state，不恢复上一轮预算。
- 报告只有来源级引用，还没有 claim 到证据片段的逐条映射。
- 搜索依赖 HTML/RSS，不执行 JavaScript。
- `run.json` 尚未记录真实 token、延迟和美元费用。
- reviewer 只能审查主 Agent 传给它的内容，没有独立读取原网页。

Stage 03 建议实现：`evidence_evaluator -> escalate_or_finish` 状态节点、动态档位升级、token/延迟/费用遥测，以及固定策略与自适应策略的基准实验。

## 11. PPT 建议页

本阶段可拆成七页：

1. Stage 01 的三个缺口：证据真假、成本失控、无法恢复。
2. 双拓扑：single 与 multi 的权限和调用链差异。
3. 四档策略：不是一个标签，而是一组硬预算和质量门槛。
4. 结构化证据：成功、失败、短页面、去重和 `[S#]`。
5. checkpoint：历史连续，但本轮 trace 独立。
6. 真实案例：reviewer 删除弱来源后触发 high 验收失败。
7. 创新演进：从静态分级到证据驱动的自适应计算。
