# TongAgent 统一评测指南

本文说明如何通过 `evaluation` 包以同一接口运行 B1、B2、B3，如何判断一次
对比是否公平，以及如何把外部研究数据集接入评测。这里的默认 fixture 运行只是
确定性 **smoke test**：它验证 runner、预算、恢复、指标和产物协议，不衡量真实
联网研究质量，也不能作为正式 benchmark 结果。

> Deterministic offline fixture smoke test; not a formal benchmark result.

主研究 CLI `search_agent.py` 适合交互式研究；正式可比较的实验应使用本页的
`python -m evaluation.cli` 入口。

## 三个 baseline 的严格定义

| ID | 实现 | 允许的能力 | 明确不包含 |
| --- | --- | --- | --- |
| B1 `simple_react` | LangChain `create_agent` 的最小 ReAct | 同一模型、共同的 `web_search` / `fetch_url`、共同预算中间件、基本终止 | ResearchPlan、Evidence Graph、Adaptive Control、研究者委派 |
| B2 `vanilla_deepagents` | 仓库固定的上游 `deepagents==0.6.12` 与 `create_deep_agent` | 同一模型和网络工具、上游原生 planning、虚拟文件系统、上游 general-purpose subagent、共同预算中间件 | TongAgent 的 ResearchPlan、Evidence Graph、Adaptive Control 和 researcher/reviewer 角色 |
| B3 `tongagent` | 生产 `search_agent.build_agent` 的完整 Stage 03D 外层图 | 显式计划、Claim–Evidence 图、结构验收、冲突状态和 adaptive controller | 主 baseline 不启用 multi；multi 必须作为单独消融实验 |

B3 主 baseline 固定为 `mode=single`、`strategy=adaptive`、
`max_escalations=2`，默认 `effort=medium`。修改前三项会被 adapter 拒绝；
`multi` 不能继续沿用 B3 主 baseline 的名称或混进同一结果表。B1/B2 虽然复用
生产搜索、抓取和指标语义包装层，却注入明确禁用 Evidence Graph 的空状态，
不会构造或写入 `EvidenceGraphStore`，也不会因此获得 TongAgent 的研究状态或
完成门槛。B1/B2 的 `evidence_count` 与
`structural_subquestion_coverage` 必须是 JSON `null`。

B3 adapter 通过依赖注入把 resolved model、共同 raw providers、预算和
middleware 交给生产 `build_agent`；它不会读取 `.env`、另建 `ChatOpenAI`，也
不会给 planner 或 reviewer 偷换模型。B2 的 parent 与原生 general-purpose
subagent 也共享同一个 model、tool 和预算 middleware 实例。

三套 adapter 都实现：

```python
run(task, resolved_config) -> RunResult
```

## 公平性合同

同一对比中的所有 `task × system` 必须共享：

- 完整数据集字节摘要、抽样 `seed` 和 backend 类型；
- fixture 模式下的 fixture 内容 revision；
- 模型 provider、模型名、采样参数、输出上限和非秘密模型参数；
- 搜索与抓取实现的身份和公共参数；
- search、fetch、全部 tool、model、token、wall time、搜索结果数、页面字符数
  和图递归深度上限；
- 外部 judge 的协议身份、版本和公共配置；没有 judge 时共同为 `null`。

完整 `config_fingerprint` 还包含 `system_id` 和各系统合法的私有选项；
`fairness_fingerprint` 只排除这两类系统特定字段和运行目录。attempt 路径不会
改变实验身份。runner 在创建目录或调用模型/工具之前预解析全部配置；如果共同
字段形成多个 fairness fingerprint，就拒绝运行。聚合器也会拒绝混合不同
fairness fingerprint 的结果。

系统 prompt、框架自带控制流和 B3 的显式研究方法本来就是被比较的处理变量，
不需要相同；它们产生的所有模型与工具调用仍受同一全局预算约束。B3 的独立
planner 调用也必须记入共同 model/token 预算与 trace，不能成为隐式免费调用。

### 共同硬预算

`resolved_config.budget` 中的限制对三个系统统一执行：

| 字段 | 含义 |
| --- | --- |
| `max_search_calls` | 被共同中间件接纳的搜索调用上限 |
| `max_fetch_calls` | 被共同中间件接纳的抓取调用上限 |
| `max_total_tool_calls` | 包含网络及框架其他工具在内的总调用上限 |
| `max_model_calls` | 所有受控模型调用上限，包括 B3 planner |
| `max_total_tokens` | 可核算模型 token 的全局硬上限 |
| `wall_time_seconds` | 单个 task/system attempt 的单调时钟期限 |
| `max_results_per_search` | 每次搜索最多交给系统的结果数 |
| `max_page_chars` | 每次抓取最多交给系统的正文字符数 |
| `recursion_limit` | 三个图共同使用的 LangGraph 递归深度上限 |

每次模型调用在 provider I/O 前先按“估算输入 token + 配置的最大输出 token”
保留额度，响应后再用 provider usage 原子结算；provider 不返回可核算 usage
时，对预算保守收取整笔 reservation，但 `token_usage` 仍保持不可用而不是补 0。
若真实 usage 超过预估 reservation，overrun 和 ceiling breach 会完整入账并阻止
后续工作，不能抹掉已经发生的 provider 消费。预算计数、provider telemetry 和
账单成本是不同概念，详见 [METRICS.md](METRICS.md)。

## JSONL 数据集

输入是 UTF-8 JSONL，每个非空行严格符合：

```json
{"id":"example-001","question":"需要研究的问题","reference_answer":"参考答案","source_dataset":"dataset/repo","source_split":"test","source_index":42,"metadata":{"split":"pilot"}}
```

规则如下：

- `id` 长度为 1–128，只能由 ASCII 字母、数字、`.`、`_`、`-` 组成，且首字符
  必须是字母或数字；同一文件中不能重复。
- `question` 必须是非空文本。
- `reference_answer` 可以为 `null`；存在时必须为非空文本。
- `source_dataset`、`source_split` 和非负整数 `source_index` 是可选的外部数据
  provenance，但必须三者同时存在或同时省略。
- `metadata` 是任意 JSON 对象，可保存原数据集 ID、split、许可证、facet、
  官方 evaluator 版本等不参与提问的元数据。
- schema 不接受上述字段以外的额外顶层字段。其他外部信息放进 `metadata`。

dataset digest 覆盖完整文件的原始字节，而不只是 `--limit` 选中的样本。
runner 使用局部 PRNG 按 `--seed` 确定性打乱，再应用 `--limit`；同一文件、
seed 和 limit 会得到相同顺序。修改文件中的空格或换行也会改变 digest，这是
有意的实验版本控制。

## CLI

以下命令都在本目录执行：

```bash
cd /home/huiwei/sy/TongAgent/langchain-deepagents/learning/search-agent
```

### 先检查调度，不创建产物

```bash
.venv/bin/python -m evaluation.cli run \
  --systems simple_react vanilla_deepagents tongagent \
  --dataset evaluation/datasets/stage_d_offline_smoke.jsonl \
  --experiment offline-smoke-dry-run \
  --seed 0 \
  --dry-run
```

`--dry-run` 会验证数据、系统、配置、fingerprint 和已有 attempt 状态，但不会
创建 attempt，也不会调用模型或工具。

### 运行确定性 fixture smoke

完整 Stage D smoke 应优先使用 canonical 脚本，并为每次运行提供全新的
experiment ID：

```bash
bash scripts/run_stage_d_smoke.sh \
  "stage-d-offline-smoke-$(date +%Y%m%d-%H%M%S)"
```

该脚本依次执行输入 validator、18 个 fresh `task × system` worker、安全 resume
及结果 hash 对比、一个保留旧 attempt 的 rerun，最后执行严格结果 validator。
它拒绝复用已有实验目录，也不会删除旧结果。只有最后输出
`Stage D deterministic fixture smoke passed` 且 validator 返回
`validation_status="passed"`，才算 Stage D smoke 验收通过。
`fresh-run.json`、`resume-run.json`、`rerun-run.json`、两份相对路径 SHA-256
清单和最终 `validation.json` 会保存在 experiment 根目录；strict validator
缺少 fresh/resume 证明时会 fail closed。

下面的命令只是直接调用底层 runner，适合单步调试；它本身不覆盖 canonical
脚本的 resume、rerun、hash 和场景级严格验收：

```bash
.venv/bin/python -m evaluation.cli run \
  --systems simple_react vanilla_deepagents tongagent \
  --dataset evaluation/datasets/stage_d_offline_smoke.jsonl \
  --experiment offline-smoke-v1 \
  --seed 0 \
  --output output/evaluations
```

默认配置使用严格 fixture backend。未知 query 或 URL 会返回
`fixture_not_found`，不会静默联网。fixture 模型、固定搜索结果和固定网页只用于
验证框架行为；即使三套系统全部完成，也不能据此声明哪套系统真实研究质量更高。

已有实验可独立运行同一个严格 validator：

```bash
.venv/bin/python -m evaluation.validate_stage_d \
  --dataset evaluation/datasets/stage_d_offline_smoke.jsonl \
  --fixtures evaluation/fixtures \
  --experiment-directory output/evaluations/<experiment-id> \
  --require-rerun
```

### 限量 pilot 与配置覆盖

仓库提供一个单题、无秘密的示例输入
`evaluation/datasets/live_pilot.example.jsonl`，以及共享 live 配置
`evaluation/configs/live_pilot.example.json`。配置中的 OpenAI endpoint 是公开
非秘密参数；凭证值仍只从父进程已有的 `OPENAI_API_KEY` 传入，不会写入 JSON。

先做零调用调度检查：

```bash
.venv/bin/python -m evaluation.cli run \
  --systems simple_react vanilla_deepagents tongagent \
  --dataset evaluation/datasets/live_pilot.example.jsonl \
  --limit 1 \
  --seed 17 \
  --output output/evaluations \
  --experiment live-pilot-dry-run \
  --config evaluation/configs/live_pilot.example.json \
  --dry-run
```

确认凭证、预算、数据授权和 experiment ID 后，正式的第一条限量 live pilot
命令是：

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

这是一题 pilot，不是正式 benchmark，也不应外推系统排名。本轮 readiness sprint
没有执行这条命令。

自定义数据和配置仍使用同一入口：

```bash
.venv/bin/python -m evaluation.cli run \
  --systems simple_react vanilla_deepagents tongagent \
  --dataset /path/to/tasks.jsonl \
  --limit 3 \
  --seed 17 \
  --experiment pilot-001 \
  --config /path/to/evaluation-config.json
```

`--config` 是对安全默认值的 JSON 深合并覆盖。`system_id`、dataset digest、
seed、artifact directory 和两个 fingerprint 由 orchestration 拥有，不能在
配置中覆盖。`system_options` 可以按系统 ID 分组，但不得用它改写共同预算。

### resume、rerun 与 no-resume

`--resume` 默认启用。仅当最新 attempt 存在合法、terminal、task/config
fingerprint 完全一致的 `result.json` 时才跳过：

```bash
.venv/bin/python -m evaluation.cli run \
  --dataset evaluation/datasets/stage_d_offline_smoke.jsonl \
  --experiment offline-smoke-v1 \
  --resume
```

要保留旧结果并显式生成下一个 attempt：

```bash
.venv/bin/python -m evaluation.cli run \
  --dataset evaluation/datasets/stage_d_offline_smoke.jsonl \
  --experiment offline-smoke-v1 \
  --rerun
```

`--no-resume` 同样不会跳过已完成项。runner 从不覆盖旧 attempt，而是递增
`attempt-0002`、`attempt-0003`。中断且没有 `result.json` 的目录会保留，下一次
运行创建新 attempt；损坏、被篡改或 fingerprint 漂移的 `result.json` 会
fail closed，不会被当作成功，也不会静默绕过。若配置或数据变化，应使用新的
experiment ID。

### 重新聚合

```bash
.venv/bin/python -m evaluation.cli summarize \
  --output output/evaluations \
  --experiment offline-smoke-v1
```

聚合器为每个 `system × task` 选择最新合法 terminal attempt，统计保留的未完成
attempt，并原子生成 `summary.json`、`summary.csv` 和 `summary.md`。系统比较
中的 **Not completed** 指所有非 `completed` terminal 状态之和，包括
`partial`、`failed`、`budget_exhausted`、`timed_out` 和 `interrupted`；它不是
`failed` 状态或 `failure_type` 的同义词。空值在 JSON 中保持 `null`，CSV 中
为空，Markdown 中显示 `—`；不会伪造为 0。

## Fresh-process 隔离

每个 `task × system` 都由全新的
`sys.executable -m evaluation.worker` 进程执行。父进程不会导入任何 adapter，
因此 B3 的模块级 harness profile、checkpoint 或全局状态不能污染 B2，某题的
模型/fixture 状态也不能泄漏到下一题。worker 输入完全来自 attempt 下的
`task.json`、`config.json` 和 `job.json`。

进程隔离不是“统计独立性”证明，也不能消除共享机器负载对 wall time 的影响。
正式耗时对比仍应固定硬件、并发、缓存和网络条件，并进行足够多的重复试验。

## 输出目录与完成协议

```text
output/evaluations/<experiment_id>/
├── summary.json
├── summary.csv
├── summary.md
├── fresh-run.json
├── fresh-results.sha256
├── resume-run.json
├── resume-results.sha256
├── rerun-run.json
├── validation.json
├── simple_react/
│   └── <task_id>/attempt-0001/
├── vanilla_deepagents/
│   └── <task_id>/attempt-0001/
└── tongagent/
    └── <task_id>/attempt-0001/
```

每个 attempt 的公共结构：

```text
attempt-0001/
├── task.json
├── config.json
├── job.json
├── worker.stdout.log
├── worker.stderr.log
├── answer.md
├── trace.json
├── metrics.json
├── failure.json
├── native/
│   ├── task.json
│   ├── resolved_config.json
│   ├── native.json
│   ├── budget.json
│   ├── tool_calls.json
│   ├── trace.jsonl
│   └── answer.md          # optional：系统确实产生 final_answer 时才有
└── result.json
```

B3 进入 TongAgent adapter 后，还会在 `native/tongagent/` 保存
`checkpoint.sqlite`、`events.jsonl`、`control.json`、`sources.json`、
`evidence.json` 和 `validation.json`。`plan.json` 只有已建立显式计划时存在，
`report.md` 只有系统确实产出报告时存在；worker 输入、adapter 构建或早期模型
调用失败时，这些 native 文件都可能不存在，失败事实以公共
`failure.json`/`result.json` 为准。

除进程启动时直接打开并持续写入的 `worker.stdout.log` 和
`worker.stderr.log` 外，runner 收尾阶段生成的 canonical 产物使用原子替换；
`result.json` 最后写入且是唯一完成标志。日志可能在中断时保留部分内容，不能
作为 attempt 已完成的判据：

- `answer.md` 是规范化最终答案视图；
- `trace.json` 是 canonical tool-call 视图；
- `native/trace.jsonl` 是有序、脱敏的 runtime 事件；
- `metrics.json` 保留指标及显式空值；
- `failure.json` 保留 terminal failure；被系统处理过的工具失败仍在 tool trace
  和 native trace 中；
- `result.json` 包含完整 `RunResult` 与 resolved config。

JSONL trace 是显式事件审计记录，不是模型隐藏思维链。当前 runner 也不承诺把
该文件逐事件实时 append；worker 的 stdout/stderr 会写入日志，完成后再以
`trace.jsonl` 审计模型调用、工具调用、预算与 B3 明示控制状态。抓取正文和密钥
不会作为完整内容写入 canonical trace；超长文本以长度和 hash 摘要代替。

## Fixture 与 live 配置的安全边界

fixture backend 必须使用 fixture 模型和空的 `model.credential_env`，worker
会移除继承环境中的模型、搜索和 tracing 凭证，并设置 offline 标志。
`fixture_revision` 是 fixture 内容摘要的一部分，也进入 fairness fingerprint。

live backend 必须显式设为 `backend_kind=live` 且
`fixture_revision=null`。凭证只能通过 `model.credential_env` 列出允许传给
worker 的**环境变量名**；变量值必须预先存在于父进程环境，绝不能写进 JSON。
resolved config、fingerprint 和 artifacts 只保存变量名，不保存值。fixture
配置携带 credential allowlist 会被拒绝；不在 allowlist 中的隐式凭证也不会
进入 worker。`credential_env` 只接受大写环境变量名，并拒绝 endpoint、
model override、tracing 和 `PYTHONPATH` 等隐式运行状态；若 live provider 需要
非秘密 `base_url`，应把它显式放入 `model.parameters`，接受它会被持久化并进入
fingerprint。

当前 live runtime 固定加载生产 `search_agent.web_search` 和
`search_agent.fetch_url`；`tools.search_backend`、`fetch_backend` 与
`parameters` 目前主要是持久化和 fingerprint 中的实现身份，不是可任选 provider
的通用 tool registry。不要只改这两个字符串就声称已经切换搜索实现。

下面只是 live 配置结构示意；仓库中的可运行示例以
`evaluation/configs/live_pilot.example.json` 为准，读取文件本身不会启动实验：

```json
{
  "backend_kind": "live",
  "fixture_revision": null,
  "model": {
    "provider": "openai",
    "name": "gpt-5.4-nano",
    "credential_env": ["OPENAI_API_KEY"],
    "temperature": 0.0,
    "max_output_tokens": 5000,
    "parameters": {}
  },
  "tools": {
    "search_backend": "tongagent-web-search",
    "fetch_backend": "tongagent-fetch-url",
    "parameters": {}
  },
  "judge": null
}
```

不要把 API key、Bearer token、cookie 或 password 写入 `parameters` 或
`system_options`；secret-like 字段会被拒绝。trace 还会对常见 secret key、
Bearer、`sk-...` 和 URL query secret 做二次脱敏，但脱敏不是把秘密写进配置的
许可。

`judge=null` 是当前默认。配置非空 judge protocol 只有在对应 judge 已在
runner registry 中实现时才可运行；未知 judge 必须 fail fast，不能把内部结构
指标伪装成 judge 分数。当前 CLI 没有内置外部 judge。由于每个 worker 是 fresh
process，仅在调用者进程临时 `register` 一个对象还不够；正式接入必须让同一
`id/version` 的实现由 orchestration preflight 和 worker 都能确定性导入。

## 内部结构指标与外部质量判断

以下指标可由 runner 自己审计：

- 调用量、期限、token telemetry 可用性、completion/failure；
- provider/nonempty/relevant/evidence-producing 搜索语义；
- 内容 revision、规范化 hostname 和保守 corroborating groups；
- B3 Claim–Evidence–Source 闭包与
  `structural_subquestion_coverage`；
- 存在短参考答案时的保守 whole-string normalized exact match。

以下结论不能由这些内部指标推出，必须依赖数据集官方 evaluator、预注册的外部
judge、人工双评或明确的可执行 validator：

- 最终答案事实正确性和宽问题的语义完整性；
- citation 是否真正蕴含 claim、来源是否权威或相互独立；
- 冲突裁决是否正确；
- 长报告的结构质量、深度、可读性和研究价值；
- 系统在真实联网分布上的总体优劣。

详尽计算和“不能代表什么”见 [METRICS.md](METRICS.md)。

## 接入外部 benchmark

仓库不捆绑 BrowseComp、完整 FRAMES、DeepResearch Bench、官方 evaluator 或
付费 judge。仓库只保存了一个从官方 FRAMES revision 确定性抽取的 5 题
pipeline pilot 子集、manifest 和选择器；它不是完整数据集或正式结果。接入其他
release 时遵循同一流程：

1. 固定数据集 release、split、许可证和原始文件 digest。
2. 写一个只做 schema 转换的 importer；每条输出
   `id/question/reference_answer/source_dataset/source_split/source_index/metadata`，
   并在 metadata 保留 release、许可证和其他 provenance。
3. 对转换后的 JSONL 计算并记录 digest，先用 `--dry-run`，再用小 `--limit`
   做 live pilot。
4. 三个系统使用同一 resolved config、seed、预算、credential allowlist 和
   judge protocol。
5. 将官方 evaluator 包装为父进程与 fresh worker 都会加载的有版本号 judge，
   或者在运行后对 `result.json.final_answer` 执行官方脚本；官方分数不能写回
   已有 attempt。
6. 保存 evaluator 版本、prompt、模型、温度、重试策略和原始 judge 输出；不应
   只保存一个浮点数。
7. 新建 experiment 运行正式全量任务，报告失败和 `null`，不要只筛选完成项。

### BrowseComp

将题目映射到 `question`，将 release 给出的参考答案映射到
`reference_answer`，把原样本 ID、split 和官方 grader 版本放入 metadata。
本项目的 normalized exact match 只能作为严格的诊断列；正式结果必须采用该
release 指定的官方评分协议。尤其不能用 substring match 或手工删前缀来抬高
准确率。

### FRAMES

保留原始多跳问题、参考答案及数据集提供的类型、来源或推理标签。不要为了适配
B3 而把同一题人工拆成多个只对 B3 可见的任务；若要评估 facet/子问题覆盖，
这些 annotations 必须作为三系统共享的外部 judge 输入。最终指标使用所选
FRAMES release 的官方 evaluator，本项目 whole-string EM 仍只是附加诊断。
当前 5 题子集的固定 revision、eligibility、seed、hash 和下一步命令见
[FORMAL_PILOT_PLAN.md](FORMAL_PILOT_PLAN.md)。

### DeepResearch Bench

长报告任务通常不能用 exact match。将研究 prompt 映射到 `question`，把 rubric、
参考材料身份和官方任务 ID 放入 metadata，并注册固定版本的 judge protocol，
分别评估答案质量、事实/引用、覆盖或其他官方维度。没有 judge 时
`judge_score` 必须为 `null`；B3 的 evidence 数或 structural coverage 不能
替代长报告质量分。

无论接入哪个数据集，fixture smoke 与 live pilot/正式 benchmark 必须使用不同
experiment ID 和报告标题。只有 `fixture_smoke=false`、官方协议固定、外部依赖
可复现且样本规模足以支持结论时，结果才可以被称为正式 baseline。
