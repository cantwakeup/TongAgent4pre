# TongAgent Benchmark Readiness Report

生成日期：2026-07-19

分支：`codex/benchmark-readiness`
基准：`agent/stage-03d-adaptive-control` / `15b3c2a`

> Deterministic offline fixture smoke test; not a formal benchmark result.

## 结论

Stage A–E 的本地实现与离线验收已经完成。当前版本具备：

- 语义可信的 search/source/structural coverage 指标；
- 可复现的离线安装与统一测试入口；
- B1/B2/B3 的共同输入、模型接口、search/fetch fixture、全局预算和
  `RunResult` 协议；
- 18 条隔离运行记录、严格聚合、failure distribution、resume/hash/rerun
  证明；
- benchmark、指标、readiness 和 pilot 操作文档。

当前可以开始**单题、严格限额的 live pilot**，但尚未运行任何 live pilot 或正式
benchmark，也不能根据本报告给 B1/B2/B3 做真实研究质量排名。正式结果仍需固定
外部数据 release、官方 evaluator 或预注册 judge、live provider 配置和足够样本。

## A/B/C/D/E 实际状态

| Stage | 状态 | 已完成证据 | 未冒充的边界 |
|---|---|---|---|
| A：指标正确性 | 完成 | `c0f559e`；搜索语义阶梯、来源 host/revision/佐证组、structural coverage、attempt/checkpoint 守恒及回归测试 | 相关性不是答案正确性；hostname 不是 publisher |
| B：可复现构建 | 完成 | `a9bdeb6`；标准 PyPI 默认、锁定依赖、本地 DeepAgents editable、`scripts/check_tongagent.sh` | 没有声称任意机器/网络都具有相同 wall time |
| C：统一 Evaluation Harness | 完成 | `6d51f71`；统一 JSONL、config/fingerprint、fresh worker、B1/B2/B3、RunResult、resume/rerun/summarize | 没有内置外部 judge；未知指标保持 `null` |
| C baseline 边界复验 | 完成 | `85a49b0` 增加运行时隔离测试；B1/B2 不构造或写入 `EvidenceGraphStore`，B3 显式启用完整 Stage 03D state | 共同 search/fetch 语义包装和硬预算属于公平评测层，不是 B1/B2 的 TongAgent 完成能力 |
| D：确定性 fixture smoke | 完成 | `85a49b0`；6 tasks × 3 systems、strict validator、tracked summary 与完整脱敏 RunResult | fixture 只验证框架，不是正式 benchmark |
| E：文档与交接 | 完成 | `BENCHMARKING.md`、`METRICS.md`、本报告、两级 README、live pilot 示例 config/dataset | 本轮没有执行文档中的 live 命令 |

## Commit 列表

本分支保留并按顺序叠加：

1. `c0f559ee4cd2c1e3ea8f312564200aec611e667d`

   `fix(metrics): correct search, source, and coverage semantics`
2. `a9bdeb6bae2e4c7776659ea6e4a2720ca94e1d68`

   `build(repo): make TongAgent tests reproducible`
3. `6d51f71715fca01fdb60ac749395953dec2da267`

   `feat(eval): add reproducible multi-system evaluation harness`
4. `85a49b04b0f115daafb3edb50f25f5851928ef31`

   `test(eval): add deterministic baseline smoke suite`
5. Stage E 为**包含本报告的提交**：

   `docs: document benchmark workflow and metric definitions`

第五个 SHA 不能写入它自身的文件内容而不改变该 SHA；请以
`git log -1 --format=%H` 或本轮最终交接回复中的值为准。前四个提交均未 amend、
rebase、reset、拆分或重写。

## 三个 baseline 的能力边界

| Baseline | 执行路径 | 拥有 | 不拥有 |
|---|---|---|---|
| B1 `simple_react` | LangChain `create_agent` 最小 ReAct | 共同 model、`web_search`、`fetch_url`、硬预算、基本终止 | Evidence Graph、Adaptive Control、TongAgent ResearchPlan、TongAgent 完成条件、委派 |
| B2 `vanilla_deepagents` | 固定 `deepagents==0.6.12` 的 `create_deep_agent` | 共同 model/tools/budget、上游 planning/filesystem/general-purpose subagent | TongAgent Evidence Graph、Adaptive Control、ResearchPlan、完成条件、researcher/reviewer |
| B3 `tongagent` | 生产 `search_agent.build_agent` | 完整 Stage 03D：计划、Evidence Graph、conflict、structural closure、adaptive controller、checkpoint | 主 baseline 不启用 multi；multi 只能作为另行命名的消融 |

B1/B2 的 `evidence_count` 与 `structural_subquestion_coverage` 在 RunResult、
JSON、CSV 和 Markdown 中均为 `null`/空/`N/A`，没有伪装成 0。B3 主 baseline
固定 `single + adaptive + max_escalations=2`。

## 公平预算规则

同一实验的三个系统必须共享并由 `fairness_fingerprint` 覆盖：

- 完整 dataset digest、seed、backend 和 fixture revision；
- model provider/name/temperature/output ceiling/公共参数；
- search/fetch 实现身份与 fixture；
- search、fetch、total tool、model、token、wall time、search result、page
  character 和 recursion ceilings；
- judge identity/version；没有 judge 时共同为 `null`。

系统 prompt、上游原生控制流和 TongAgent 方法是被比较的处理变量，不要求相同；
但它们产生的模型和工具调用都必须进入同一硬预算。B3 planner 与 DeepAgents
summarization 也不能获得未记录的免费模型调用。聚合器拒绝 mixed fairness
fingerprint 或 mixed git SHA。

## Stage D post-commit 运行 provenance

- Experiment：`stage-d-offline-smoke-postcommit-20260719-a9`
- Git SHA：`85a49b04b0f115daafb3edb50f25f5851928ef31`
- Dataset digest：
  `sha256:b0c8a12630cb71416009128d2f6bd1703a5418ef7926e1807ac73dc71a245487`
- Fixture revision：
  `6f09f8f239d324b03ff8f948e8dfb29dfeacdf3e24c20d146514a06b742b99e9`
- Fairness fingerprint：
  `sha256:d0609f6df2bfa994785e85f525398268037f3d2c1f6a7537685f48c499cec4b8`
- Fresh：18 executed / 0 skipped
- Resume：0 executed / 18 skipped
- Resume 前后：18 个 terminal `result.json` SHA-256 全部不变
- Targeted rerun：1 executed，保留 `attempt-0001` 并新增 `attempt-0002`
- Strict validator：`validation_status=passed`
- Incomplete attempts：0

仓库内跟踪的脱敏导出位于
`evaluation/results/stage_d_offline_smoke/`。它来自提交前 canonical a8，并明确
记录 source Git SHA `6d51f71`；提交后 a9 使用完全相同 dataset、fixture、
fairness fingerprint 和预期语义，在 Stage D commit `85a49b0` 上再次通过。
本机完整输出保留在被 Git 忽略的 `output/evaluations/`，没有提交缓存、SQLite、
密钥或大型网页正文。

## 18-run 离线汇总

以下 wall time 来自同机单次 deterministic fixture smoke，只用于执行成本烟测。
`N/A` 是 B1/B2 不拥有的 TongAgent 指标，不是 0。

| System | Task | Status | Failure | Tools | Wall (s) | Evidence | Structural coverage |
|---|---|---|---|---:|---:|---:|---:|
| simple_react | fixture-budget-resume | budget_exhausted | budget_exhausted | 5 | 0.698082 | N/A | N/A |
| simple_react | fixture-conflict | completed | — | 5 | 0.694414 | N/A | N/A |
| simple_react | fixture-fetch-retry | completed | — | 4 | 0.688094 | N/A | N/A |
| simple_react | fixture-nonempty-irrelevant | completed | — | 1 | 0.663944 | N/A | N/A |
| simple_react | fixture-one-hop | completed | — | 3 | 0.671405 | N/A | N/A |
| simple_react | fixture-two-source | completed | — | 4 | 0.686637 | N/A | N/A |
| vanilla_deepagents | fixture-budget-resume | budget_exhausted | budget_exhausted | 5 | 0.756119 | N/A | N/A |
| vanilla_deepagents | fixture-conflict | completed | — | 5 | 0.762934 | N/A | N/A |
| vanilla_deepagents | fixture-fetch-retry | completed | — | 4 | 0.738277 | N/A | N/A |
| vanilla_deepagents | fixture-nonempty-irrelevant | completed | — | 1 | 0.715323 | N/A | N/A |
| vanilla_deepagents | fixture-one-hop | completed | — | 3 | 0.732620 | N/A | N/A |
| vanilla_deepagents | fixture-two-source | completed | — | 4 | 0.746790 | N/A | N/A |
| tongagent | fixture-budget-resume | budget_exhausted | budget_exhausted | 6 | 1.936910 | 0 | 0.0 |
| tongagent | fixture-conflict | completed | — | 10 | 2.523870 | 3 | 1.0 |
| tongagent | fixture-fetch-retry | completed | — | 8 | 2.141800 | 2 | 1.0 |
| tongagent | fixture-nonempty-irrelevant | partial | — | 2 | 1.700060 | 0 | 0.0 |
| tongagent | fixture-one-hop | completed | — | 7 | 1.794050 | 2 | 1.0 |
| tongagent | fixture-two-source | completed | — | 8 | 1.817440 | 2 | 1.0 |

### 每系统汇总

| System | Runs | Completed | Partial | Budget exhausted | Total tools | Mean tools | Total wall (s) | Mean wall (s) | Failure distribution |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| simple_react | 6 | 5 | 0 | 1 | 22 | 3.66667 | 4.10258 | 0.683763 | budget_exhausted: 1 |
| vanilla_deepagents | 6 | 5 | 0 | 1 | 22 | 3.66667 | 4.45206 | 0.742010 | budget_exhausted: 1 |
| tongagent | 6 | 4 | 1 | 1 | 41 | 6.83333 | 11.91412 | 1.985687 | budget_exhausted: 1 |

### 场景语义验收

- 非空无关搜索：三系统均保存
  `provider_success=true`、`nonempty_search=true`、
  `relevant_search=false`；TongAgent 为 honest `partial`，0 Evidence、0.0
  structural coverage，没有正常完成研究。
- 多来源：Stage D 两个来源形成 2 个 host、2 个 captured revision、2 个保守
  corroborating groups；专项测试另外证明同 host 不自动算两组、不同 host 的
  exact mirror 也不自动算两组。
- 冲突：TongAgent 保存一个 `unresolved` conflict，Claim 状态为 `contested`，
  报告将该 Claim 放在 `Conflicts and Caveats`，没有静默消解。
- 抓取失败：首次响应保存 `failure_class=timeout`、`retryable=true`，同 URL
  第二次成功；失败调用仍消费 fetch allowance，三系统均记录 3 次 fetch，没有
  返还后重复计数。
- 预算耗尽：三系统均为 `completion_status=budget_exhausted` 和
  `failure_type=budget_exhausted`，没有伪装为 `completed`。
- resume：第二次跳过 18 对 system/task，计数与 summary 不变；显式 rerun
  只新增指定 pair 的 attempt。

## 测试命令与准确结果

所有命令均未访问公网、未调用真实模型或付费 API；pytest 使用 socket 禁用插件。

```bash
cd learning/search-agent
.venv/bin/python -m pytest -q --disable-socket --allow-unix-socket \
  tests/test_evaluation_stage_d.py
```

结果：`6 passed, 0 failed, 0 skipped`。

```bash
.venv/bin/python -m pytest -q --disable-socket --allow-unix-socket \
  tests/test_evaluation_*.py
```

结果：`80 passed, 0 failed, 0 skipped`。

```bash
.venv/bin/python -m pytest -q --disable-socket --allow-unix-socket tests
```

结果：`229 passed, 24 subtests passed, 0 failed, 0 skipped`。

```bash
cd ../..
UV_OFFLINE=1 \
UV_BIN=/home/huiwei/miniconda3/envs/wzq_base/bin/uv \
bash scripts/check_tongagent.sh
```

结果：

- locked sync：68 packages resolved，66 checked，offline；
- format：47 files already formatted；
- lint：passed；
- compile/import：passed；
- benchmark-metric regressions：`82 passed, 21 subtests passed`；
- complete offline suite：`229 passed, 24 subtests passed`；
- gate：`TongAgent checks passed`。

Stage E 的 live pilot 示例只做了零调用调度验证：

```bash
.venv/bin/python -m evaluation.cli run \
  --systems simple_react vanilla_deepagents tongagent \
  --dataset evaluation/datasets/live_pilot.example.jsonl \
  --limit 1 --seed 17 \
  --output /tmp/tongagent-live-pilot-dry-run \
  --experiment live-pilot-doc-check \
  --config evaluation/configs/live_pilot.example.json \
  --dry-run
```

结果：`dry_run=3`、`executed=0`、`skipped=0`；未创建 attempt，未调用 model 或
search/fetch provider。

Stage A 已接受 checkpoint 的专项证据为：

| 专项命令（共同使用 `-q --disable-socket --allow-unix-socket`） | 结果 |
|---|---|
| `pytest tests/test_metric_semantics.py` | 25 passed，10 subtests |
| `pytest tests/test_evidence_graph.py` | 24 passed |
| `pytest tests/test_adaptive_control.py tests/test_adaptive_graph.py` | 23 passed，11 subtests |
| `pytest tests/test_checkpoint.py` | 7 passed |

没有删除、跳过或弱化旧测试以换取通过。

## 修改面

- Stage A：`search_agent.py`、Evidence/Research/Telemetry 模块、Stage 03D 文档及
  指标回归测试。
- Stage B：依赖配置、锁文件、根验证脚本、README/CI。
- Stage C：`evaluation/` contract、config、budget、offline providers、fresh
  execution、aggregation、B1/B2/B3 adapters 与测试。
- Stage D：dataset、9 个页面 fixture、8 个 search fixture、strict validator、
  canonical smoke 脚本、baseline 隔离测试和 tracked sanitized results。
- Stage E：`BENCHMARKING.md`、`METRICS.md`、本报告、根 README、search-agent
  README、live pilot 示例 dataset/config。

## 未运行的正式实验与外部依赖

本轮明确没有：

- 调用真实模型、付费 API 或公网搜索；
- 运行 live pilot 或任何正式 benchmark；
- 下载或捆绑 BrowseComp、FRAMES、DeepResearch Bench 数据；
- 注册外部 judge 或运行官方 evaluator；
- push、merge 或创建 PR。

正式 benchmark 仍需用户确认数据许可证/release、provider 与凭证、官方评分协议、
硬件/并发、网络/缓存条件、重复次数和费用上限。

## 已知风险与技术债务

1. `search_agent.py` 当前约 4,675 行，职责仍过多，是明确技术债务。本轮按要求
   未重构；后续应把 CLI/orchestration、artifact serialization、provider
   wrappers 和 validation 分模块，但不能在 pilot 前做无关大改。
2. live evaluation 尚未实际验证 provider/model/tool 兼容性、凭证传递、网络
   波动、限流或真实 token telemetry。
3. 当前 live runtime 固定加载生产 search/fetch；配置中的 tool backend 字符串
   主要是 identity/fingerprint，不是通用 provider registry。
4. `structural_subquestion_coverage` 只证明结构闭包，不证明答案准确、语义完整或
   citation entailment。
5. hostname 分组不是注册域/publisher 识别；不同 host 仍可能同源。
6. 外层 checkpoint 不能保证内层 Deep Agent 网络调用 exactly once；进程在内层
   节点中断时可能重跑该节点。
7. fixture wall time 与近似 token usage 不能外推真实 provider 成本或延迟。
8. 当前无外部 judge；长报告质量与真实 benchmark 分数仍不可用，必须保持
   `null`。

## 下一步正式 pilot

先检查 `evaluation/configs/live_pilot.example.json` 的 provider、model、endpoint、
预算与费用授权，并在父进程环境中设置 `OPENAI_API_KEY`。然后从
`learning/search-agent` 运行：

```bash
test -n "${OPENAI_API_KEY:-}" &&
.venv/bin/python -m evaluation.cli run \
  --systems simple_react vanilla_deepagents tongagent \
  --dataset evaluation/datasets/live_pilot.example.jsonl \
  --limit 1 \
  --seed 17 \
  --output output/evaluations \
  --experiment "live-pilot-$(date +%Y%m%d-%H%M%S)" \
  --config evaluation/configs/live_pilot.example.json
```

这条命令会产生 3 条 live RunResult，仍只是第一轮 pilot，不是正式 benchmark。
运行前建议先加 `--dry-run` 使用固定 experiment ID 验证调度；正式运行必须换成
全新 ID，不能把 dry-run 或 fixture 目录复用为 live 结果。
