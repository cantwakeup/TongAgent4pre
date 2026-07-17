# TongAgent 真实联网研究 Agent

这是一个建立在 LangChain、LangGraph 和 Deep Agents 之上的可运行研究 Agent。它会先生成显式研究计划，再按子问题真实搜索网页、抓取证据、登记经过原文子串校验的 Claim-Evidence 图、计算结构覆盖度并生成带 Claim 与来源编号的报告，同时保存计划状态、工具轨迹、来源修订、证据图和 checkpoint。

## 当前闭环

```text
用户问题
  -> 硬资源上限（effort）、预算策略（fixed/adaptive）与初始拓扑路由（mode）
  -> 结构化 research plan：SQ1 ... SQn
  -> fixed：一次性切完每个 SQ 的固定额度
     或 adaptive：每个 SQ 先获 1 search / 1 fetch，其余留作全局 reserve
  -> 外层图：select -> research -> evaluate
     adaptive 时再进入 deterministic control -> select/report
  -> single：内层主 Agent 直接搜索和读取
     或 multi：内层主 Agent -> researcher -> 可选 reviewer
  -> DDG 低相关或失败时自动查询 Bing
  -> canonical 来源账本 [S1] [S2] ...
  -> 模型提交 claim + exact quote；代码对规范化网页正文做严格子串校验
  -> Claim [C#] -> Evidence [E#] -> Source [S#]；相反 stance 形成 Conflict [X#]
  -> schema v1 Claim 闭包、`successful_searches`、revision 独立来源和 multi 委派硬门槛
  -> write_file 写入严格四节的 [C#][S#] 报告
  -> 图完整性、逐行 Claim/来源映射和 deterministic Sources 验收
  -> run-scoped report/evidence/plan/events/control/trace/sources/run + SQLite checkpoint
```

Agent 没有 shell 工具。它只能访问公开 HTTP(S) 页面，并只能写入本次 run 的
隔离虚拟根；对应 host 路径位于 `output/runs/`。

## 环境与安装

要求 Python `>=3.11,<4.0`，不需要 CUDA、本地 GPU 或 Conda。在本目录执行：

```bash
../../../.tools/uv sync
cp .env.example .env
```

然后在 `.env` 中填写私有配置：

```text
SEARCH_AGENT_BASE_URL=<private-api-base-url>
SEARCH_AGENT_API_KEY=<private-api-key>
```

`.env`、`.venv/` 和 `output/` 都不会进入 Git。

## 最简单的运行

```bash
.venv/bin/python search_agent.py --effort low --mode single "Python 3.13 有哪些主要变化？"
```

默认实时显示模型文本、工具调用和工具结果大小，但不打印网页正文。加入 `--no-stream` 可关闭实时输出，加入 `--print-report` 可在验收后打印完整报告。

## 可选 Agent 拓扑

`--mode` 控制由谁完成研究：

| mode | 行为 |
| --- | --- |
| `single` | 主 Agent 直接拥有 `web_search` 与 `fetch_url`，不提供 `task` |
| `multi` | 主 Agent 不拥有网络工具或 `record_evidence`；必须由 `task(researcher)` 搜索、抓取并登记证据 |
| `auto` | `high/xhigh` 或复杂、较长问题走 multi，其余走 single |

在需要审稿的档位，multi 路径为：

```text
主 Agent -> task(researcher) -> 搜索、抓取、record_evidence
        -> parent 只读 get_evidence_graph / update_subquestion
        -> task(reviewer)   -> 按 canonical Claim 和 exact quote 修正问题
        -> write_file       -> 严格 [C#][S#] 报告
```

`auto` 只在运行开始时把请求确定性解析为 single 或 multi。即使选择
`--strategy adaptive`，运行中的模型和 topology 也保持固定；03D 控制器只在
当前 effort 的硬上限内释放搜索/抓取 reserve。

## 四档研究强度

`--effort low|medium|high|xhigh` 不只是更换模型，还会改变硬工具预算、证据门槛、页面长度、输出上限和 reviewer 要求：

| effort | 默认主模型 | 最多子问题 | 搜索/抓取上限 | 最少成功来源 | multi reviewer |
| --- | --- | ---: | ---: | ---: | --- |
| `low` | `gpt-5.4-nano` | 1 | 2 / 3 | 2 | 否 |
| `medium` | `gpt-5.4-nano` | 2 | 4 / 6 | 2 | 否 |
| `high` | `gpt-5.4-mini` | 4 | 8 / 9 | 3 | 是 |
| `xhigh` | `gpt-5.4-mini` | 5 | 12 / 14 | 4 | 是 |

可用 `--model` 覆盖主模型，`--worker-model` 覆盖 reviewer；reviewer 默认使用免费的 `deepseek-v4-flash`。

本 API 网关的免费模型有明确兼容性边界：Qwen 当前不能自动选择工具；DeepSeek 的简单工具往返可用，但在完整 Deep Agents harness 中不够稳定。因此免费 DeepSeek 只承担无工具 reviewer，主 Agent 的工具循环仍使用 nano/mini。离线单元测试不会消耗 API token。

## fixed 与 adaptive 预算策略

`--effort` 始终是整份计划的硬 ceiling，不是 adaptive 的起点。控制器不会从
low 自动升到 medium，也不会超过该档的 `max_searches/max_fetches`：

| strategy | 初始 SQ 额度 | 剩余预算 | `evaluate` 之后 |
| --- | --- | --- | --- |
| `fixed`（默认） | 把 effort 的全部搜索/抓取预算确定性分给各 SQ | 无控制器 reserve | 沿用 03C 的有界循环 |
| `adaptive` | 每个 SQ 先获得 `1 search / 1 fetch` | effort 总上限减去 baseline | 根据证据缺口继续、释放 reserve、停止或结束 |

adaptive 控制器只读取结构化状态，不再调用一个模型来决定策略。它会综合
Claim、独立来源、成功搜索、Evidence 完整性、当前周期的新证据、工具失败和
剩余额度，记录 `continue`、`expand_budget`、`stop_subquestion`、
`finish_success`、`finish_partial` 或 `fail_closed`。每次
`expand_budget` 最多为当前 SQ 释放一个搜索和一个抓取额度；实际增量取决于
缺口类型及剩余 reserve。`--max-escalations` 限制整份 plan 可发生的释放次数，
重复恢复同一个 controller decision 不会重复加预算。

运行时和恢复时使用同一组 grant 不变量：每个 SQ 从 `1/1` baseline 开始，每次
decision 每维最多 `+1`；grant ID 顺序、目标 SQ、预算 before/after、reserve
守恒和最终 ledger 必须全部一致。checkpoint 计数只接受真实整数，拒绝布尔、
小数、负数、越界值以及与当前 plan 不一致的 budget scope。
每次成功 grant 还会持久化目标、added 和原始跃迁；若异常发生在 grant side
effect 之后、controller checkpoint 之前，同进程 replay 会返回首次记录并保持
审计 history 一致。

下面是一条中文机制验收模板；它会创建独立的 run 目录。该命令只是复现入口，
是否通过仍以本机联网运行生成的 `run.json.validation` 为准：

```bash
.venv/bin/python search_agent.py \
  --strategy adaptive \
  --max-escalations 2 \
  --effort medium \
  --mode multi \
  --model gpt-5.4-nano \
  --worker-model deepseek-v4-flash \
  --thread-id "stage03d-bigai-$(date +%Y%m%d-%H%M%S)" \
  --no-stream \
  --print-report \
  "这是 Stage 03D 的窄范围机制验收，只建立两个原子子问题。SQ1 核验北京通用人工智能研究院官网 about 页所述机构性质；SQ2 核验 https://www.bigai.ai/tongprogram-2026/ 公开的通计划联系邮箱。每个 SQ 必须委派 researcher、至少搜索一次、抓取对应官网页并登记 exact quote；若 baseline 不足，让外层 adaptive controller 根据证据缺口释放 reserve。不得研究论文、项目或产业合作，也不得把其他含“通用”的机构混入。"
```

最终 release smoke 已以 `low/single/adaptive` 对两个 Python 官方页面跑通：
`validation=passed`、`coverage=1.0`、2 个来源、2 个 Claim、2 个 Evidence、
`escalation_count=1`，decision 序列为
`continue -> expand_budget -> finish_success`。对应产物位于
`output/runs/stage03d-release-final-20260717-a1-12314473/20260717T051353.769171Z-e343e56c/`。

adaptive 不会自动改写 query。搜索词仍由研究模型在下一轮选择；代码只把每次
网络工具调用记入有序 attempt ledger。每条 `A#` 保存 SQ、工具、query/URL、
outcome、`failure_class`、`retryable`、原始 status 和短 error。当前分类包括
`none`、`budget`、`safety`、`content`、`network`、`provider`、`http` 和
`unknown`；被预算拒绝的调用也会留下 attempt，便于区分“没有尝试”“尝试失败”
和“工具成功但结果低相关”。

## checkpoint 与多轮追问

默认 checkpoint 位于 `output/checkpoints.sqlite`。不传 `--thread-id` 时，每次生成新的 UUID，避免意外串线：

```bash
.venv/bin/python search_agent.py --effort medium "研究 LangGraph checkpoint"
```

显式复用同一个 ID 即可恢复历史：

```bash
.venv/bin/python search_agent.py --thread-id demo-01 "研究 LangGraph checkpoint"
.venv/bin/python search_agent.py --thread-id demo-01 "基于前面的内容再解释 thread_id"
```

一次进程内也可以连续追问：

```bash
.venv/bin/python search_agent.py \
  --thread-id demo-02 \
  --follow-up "比较 SQLite 与内存 checkpoint" \
  "先解释 LangGraph checkpoint"
```

用 `--checkpoint-db /path/to/research.sqlite` 可指定数据库。恢复线程时，模型能看到旧消息；如果 `checkpoint.next` 表明图仍有待执行节点，CLI 会先用 `invoke(None)` 精确继续旧节点，再接收真正的新问题。预算计数、来源 ID、来源修订元数据、`C#/E#/X#`、exact quote/hash 和事件都会从 checkpoint 恢复；新生成的 `trace.json` 仍只记录本次 CLI 进程新增的消息。

当前恢复保证在外层子问题节点边界生效。内层 Agent 返回时，来源账本与证据图 snapshot 会和节点结果一起写入外层状态；如果进程恰好终止在网络工具内部，该工具节点仍可能重跑。未完成 plan 会恢复剩余硬预算；已完成 thread 的新 plan 会保留累计来源目录和连续 `[S#]` 编号，但重置 plan-local Claim 图与搜索/抓取计数。

03D 还会 checkpoint `adaptive_control`、attempt ledger、完整 grant record 和
controller decision history。未完成 plan 恢复时，CLI 对 strategy、effort、
请求 mode、解析后的 topology、主/worker 模型及 `max-escalations` 计算
policy fingerprint（fixed 的 effective max escalation 固定为 0）；任一有效项
漂移都会拒绝继续，避免用另一套语义解释已消费预算。
旧 checkpoint 若没有 controller state，只能按 `strategy=fixed` 恢复。
恢复前还会要求 plan 的 SQ ID 与 budget limits/usage/active scope 精确一致；
checkpoint 若停在首个 controller decision 之前，可以用空 decision history
恢复，但必须仍是零 grant 的纯 baseline 状态。

checkpoint 不保存整页正文。正文只在抓取后的当前进程内规范化缓存，用于 `record_evidence` 的 exact-quote 子串校验；恢复后已有 `C#/E#/X#` 仍可审计，但若要登记新的 Evidence，必须重新抓取该 `[S#]` 的 canonical URL。相同 URL 内容变化时不会覆盖旧 hash：`content_revisions` 保留每次已见修订的 hash、标题、字符数和证据质量，Evidence 绑定其抓取时的 `source_content_sha256`。

新计划使用 `evidence_schema_version=1`。`covered` 现在要求 Claim-Evidence-Source 闭包：当前 SQ 至少有一个 `supported` 或 `contested` canonical Claim，提交的 `[S#]` 必须与这些 Claim 的 Evidence 边完全一致。它仍是 claim-backed 结构状态，不等于代码已经证明 claim 的语义真伪或完整覆盖了一个宽问题；`supports/contradicts` stance 仍由模型标注。Stage 03B 的旧 checkpoint 会按 schema v0 兼容读取，不会凭空补造 Claim。

每个 plan 创建后，代码会按 SQ 数量确定性切分搜索和抓取额度。例如 medium 的两个 SQ 各自保留 `2 search / 3 fetch`，SQ1 不能消费 SQ2 的份额；checkpoint 会同时恢复当前 SQ、各 SQ 上限和已经使用的次数。进入最终报告阶段后不再激活任何 SQ，因此网络工具不能继续消耗研究预算。

搜索结果会携带 `engine` 与 `relevance_score`。当 DuckDuckGo 失败、为空，或相关结果少于两条时，代码自动查询 Bing，并在结果中记录 `status`、`engine_status`、`fallback_reason`、`search_quality` 和 `relevant_results`。只有至少一个 provider 正常返回的调用才增加 `successful_searches`；DuckDuckGo 与 Bing 都异常时返回 `status=error`，该尝试仍消耗搜索预算但不满足成功搜索门槛。相关性分数只用于发现明显噪声和触发备用引擎，不等于语义相关性证明。

## 输出与验收

每次 CLI 调用都会先创建互不覆盖的目录：

```text
output/runs/<sanitized-thread-prefix>-<thread-hash>/<UTC-run-id>/
```

同一 thread 的多次运行位于同一个 thread-hash 目录下，但各自拥有新的
`UTC-run-id`。thread ID 会先清洗并附加 hash，不能借 `../` 逃离 `runs/`。
SQLite checkpoint 默认仍位于共享的 `output/checkpoints.sqlite`。每个 run
目录生成：

| 文件 | 内容 |
| --- | --- |
| `<run-dir>/report.md` | 最终研究报告 |
| `<run-dir>/plan.json` | 当前显式计划、子问题状态、结构覆盖度与预算 |
| `<run-dir>/events.jsonl` | 计划、覆盖评估和控制事件；03D 新增 `control_assessed`、`budget_expanded`、`adaptive_continued`、`adaptive_stopped`、`control_integrity_failed` |
| `<run-dir>/trace.json` | 本次主图工具调用及结果 status/phase，不含网页正文和密钥 |
| `<run-dir>/sources.json` | 搜索/抓取预算、attempt ledger、来源修订、失败来源与当前证据图 snapshot |
| `<run-dir>/evidence.json` | 被 Evidence 使用的来源、canonical Claims、exact quotes、冲突和完整性错误 |
| `<run-dir>/control.json` | 完整 controller state：策略指纹、硬 effort、固定模型/topology、assessment、reason codes、decision history 和预算前后值 |
| `<run-dir>/run.json` | 输出目录、模型、拓扑、effort/strategy、thread、预算/attempt ledger、adaptive 摘要、委派角色、基础 token 用量、证据图计数和验收结果 |
| `output/checkpoints.sqlite` | 可恢复的消息、计划、事件、预算、来源修订与 `C#/E#/X#`；不含整页正文 |

抓取成功不等于合格证据：少于 300 个可见字符的页面标记为 `insufficient_content`；300–499 字符的短页只有在同一 host 已存在至少一条 500 字符以上的完整证据时，才以 `evidence_quality=limited` 入账；500 字符以上为 `full`。URL fragment 会被去重，403/404、超时和安全拒绝会作为失败来源记录。每个 canonical URL 保留全部 `content_revisions`；正文相同的不同 URL 会在最新来源快照中标记 `duplicate_of_source_id`。这个 latest duplicate alias 可随重抓正文变化而重算；历史 plan 的“独立来源”门槛不读取该易变字段，而是按该 plan 的 Evidence 边所绑定的 `source_content_sha256`（来源 revision）去重计算。

研究角色可以通过 `get_source_ledger` 读取 canonical `[S#]`，并用 `get_evidence_graph` 读取 `C#/E#/X#`。single 模式由主 Agent 调用 `record_evidence`；multi 模式只把该工具交给 researcher，parent 只能读取证据图并更新 SQ，不能自行登记 Evidence。调用 `record_evidence(source_id, claim, quote, stance)` 时，`claim` 必须是 12–500 个规范化字符的完整命题，`quote` 必须是 12–800 个字符且严格存在于当前进程缓存的规范化网页正文中；新建 Claim 时省略 `claim_id`，代码分配编号。代码保存 quote/hash 和来源修订 hash，但 `supports` 或 `contradicts` 是模型对边的标注，不是自动事实证明。同一 Claim 同时出现支持与反驳 Evidence 时，代码建立一个 `unresolved` 的 `[X#]`，但不自动裁决哪一方正确。

schema v1 还执行四组运行时硬门槛：使整个 plan 完成的 covered 更新必须满足计划所需 `successful_searches` 和 effort 的独立来源数，失败但已消耗预算的搜索尝试不计入成功搜索门槛；multi 模式每个 SQ 的有效委派必须位于当前 `research-step-*` 之后，是 description 以正确 `[SQ:<active-id>]` 开头的 `task(subagent_type="researcher")`，并已收到与 tool-call ID 匹配的成功 `ToolMessage`，未返回、失败、错误 SQ 或旧步骤调用均不算；任一 active SQ 尝试 blocked 时，只要搜索或独立来源 policy gap 仍可恢复且该 SQ 对应的已授予额度或 adaptive reserve 尚可补足，就拒绝 premature blocked；CLI 会重新计算整个 Claim-Evidence-Source/Conflict 闭包。

high/xhigh multi 的 reviewer 门槛也要求“成功发生在最终
`write_file` 之前”：只有 report phase 中 tool-call ID 匹配且
`ToolMessage.status=success`、结果非空的 `task(reviewer)` 才算。调用失败、
没有返回，或者最后一次 `write_file` 之后才审稿，都会使验收失败。

最终报告由显式独立、无研究工具的 synthesis-only agent 生成；内部 graph API
若没有提供该 agent 会直接拒绝构建，不会隐式回退到 research agent。报告只允许
四个 H2，顺序固定为 `Short Answer`、`Key Findings`、`Conflicts and Caveats`、
`Sources`；H1 或任何额外 heading 都会被拒绝。每条可报告事实必须逐字复制一个
`claim.text`，严格写成 `- <exact claim.text> [C#][S#]`；contested Claim 只能
进入冲突节并同时列出支持与反驳来源，contradicted-only Claim 不能作为事实。
无 C/S 的 caveat 只能逐字复制代码根据 plan/integrity 状态生成的白名单行，任意
自由文本都会被拒绝。canonical/cited `[S#]` 只从含一个 `[C#]` 的受约束事实行
提取，标题或 Sources 自报的编号不能补足引用门槛。程序据此重写 deterministic
Sources；该节每个非空行都必须精确为
`- [S#] <canonical title> — <canonical URL>`，无 bullet 的事实、额外 prose、
未知或错配编号都会被拒绝。

## 离线回归

```bash
.venv/bin/python -m unittest discover -s tests -v
```

当前 116 项全量离线测试全部通过。除原有拓扑、搜索、Evidence Graph、checkpoint
和严格报告协议外，03D 覆盖了纯函数控制决策、完整性失败 fail-closed、baseline
与 reserve、单调且幂等的硬上限 grant、attempt 分类、policy drift 拒绝、控制器
节点前恢复只应用一次 grant、grant/checkpoint crash-window 原跃迁 replay、逐步
grant/ledger 审计、严格整数与 plan/budget
scope 恢复、首个 decision 前的合法恢复、独立 synthesis agent、caveat 白名单、
reviewer 必须成功且早于最终写入、trace status/phase，以及不同 thread/run 的
输出目录隔离。

`run.json.api_usage` 汇总当前 plan checkpoint 中 AI 消息的 provider-reported input/output/cache-read token。结构化 planner 调用尚未进入消息状态，multi 模式的嵌套 subagent 用量也可能不进入外层消息；供应商没有返回账单或价目表，所以当前明确记录 `planner_call_included=false` 和 `estimated_cost_usd=null`，不会用猜测价格冒充真实费用。

## 安全边界

`fetch_url` 只允许公网 HTTP(S)，拒绝本机与内网地址，逐跳校验重定向，并限制重定向次数、下载字节数和返回字符数。网页正文和 exact quote 都被当作不可信数据；quote/hash 只能证明“本轮登记的字符串来自本轮抓取并且未被静默改写”，不能证明发布者身份、页面真实性、时效性或 claim 的语义蕴含关系。

这仍然不是浏览器级安全沙箱。生产系统还需要更强的网络隔离、DNS rebinding 防护、域名策略、提示注入检测、人工审批、全局费用控制和可观测性。

另外，03D 的恢复仍以外层图节点为原子边界：内层 Deep Agent 没有自己的
checkpoint。如果进程在内层网络工具执行中崩溃，外层尚未提交该节点，恢复后
可能重跑整个内层节点和其中的网络调用；当前的 grant ID 幂等不能提供所有外部
工具的 exactly-once 保证。

coverage 也仍是 SQ 级 Claim-Evidence 闭包，不是问题 facets 的语义清单。一个
宽 SQ 中若同时要求“成立背景、团队、项目和进展”，代码尚未把每个字段拆成必须
逐项满足的机器可读 facet；控制器只看 Claim、来源、成功搜索、失败类型和预算
缺口。因此应继续把验收问题写成原子 SQ，宽问题即使结构 coverage 达标也仍需
人工或后续 semantic-facet validator 检查内容完整性。
