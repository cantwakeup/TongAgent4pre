# TongAgent 真实联网研究 Agent

这是一个建立在 LangChain、LangGraph 和 Deep Agents 之上的可运行研究 Agent。它会先生成显式研究计划，再按子问题真实搜索网页、抓取证据、登记经过原文子串校验的 Claim-Evidence 图、计算结构覆盖度并生成带 Claim 与来源编号的报告，同时保存计划状态、工具轨迹、来源修订、证据图和 checkpoint。

## 当前闭环

```text
用户问题
  -> 资源策略（effort）与拓扑路由（mode）
  -> 结构化 research plan：SQ1 ... SQn
  -> 将搜索/抓取预算切成每个 SQ 的保留额度
  -> 外层图：select -> research -> evaluate -> loop
  -> single：内层主 Agent 直接搜索和读取
     或 multi：内层主 Agent -> researcher -> 可选 reviewer
  -> DDG 低相关或失败时自动查询 Bing
  -> canonical 来源账本 [S1] [S2] ...
  -> 模型提交 claim + exact quote；代码对规范化网页正文做严格子串校验
  -> Claim [C#] -> Evidence [E#] -> Source [S#]；相反 stance 形成 Conflict [X#]
  -> schema v1 Claim 闭包、`successful_searches`、revision 独立来源和 multi 委派硬门槛
  -> write_file 写入严格四节的 [C#][S#] 报告
  -> 图完整性、逐行 Claim/来源映射和 deterministic Sources 验收
  -> report/evidence/plan/events/trace/sources/run + SQLite checkpoint
```

Agent 没有 shell 工具。它只能访问公开 HTTP(S) 页面，并只能在 `output/` 虚拟根中写文件。

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

`auto` 目前只在运行开始时做确定性路由，还不是根据中途证据动态升级的完整控制器。

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

checkpoint 不保存整页正文。正文只在抓取后的当前进程内规范化缓存，用于 `record_evidence` 的 exact-quote 子串校验；恢复后已有 `C#/E#/X#` 仍可审计，但若要登记新的 Evidence，必须重新抓取该 `[S#]` 的 canonical URL。相同 URL 内容变化时不会覆盖旧 hash：`content_revisions` 保留每次已见修订的 hash、标题、字符数和证据质量，Evidence 绑定其抓取时的 `source_content_sha256`。

新计划使用 `evidence_schema_version=1`。`covered` 现在要求 Claim-Evidence-Source 闭包：当前 SQ 至少有一个 `supported` 或 `contested` canonical Claim，提交的 `[S#]` 必须与这些 Claim 的 Evidence 边完全一致。它仍是 claim-backed 结构状态，不等于代码已经证明 claim 的语义真伪或完整覆盖了一个宽问题；`supports/contradicts` stance 仍由模型标注。Stage 03B 的旧 checkpoint 会按 schema v0 兼容读取，不会凭空补造 Claim。

每个 plan 创建后，代码会按 SQ 数量确定性切分搜索和抓取额度。例如 medium 的两个 SQ 各自保留 `2 search / 3 fetch`，SQ1 不能消费 SQ2 的份额；checkpoint 会同时恢复当前 SQ、各 SQ 上限和已经使用的次数。进入最终报告阶段后不再激活任何 SQ，因此网络工具不能继续消耗研究预算。

搜索结果会携带 `engine` 与 `relevance_score`。当 DuckDuckGo 失败、为空，或相关结果少于两条时，代码自动查询 Bing，并在结果中记录 `status`、`engine_status`、`fallback_reason`、`search_quality` 和 `relevant_results`。只有至少一个 provider 正常返回的调用才增加 `successful_searches`；DuckDuckGo 与 Bing 都异常时返回 `status=error`，该尝试仍消耗搜索预算但不满足成功搜索门槛。相关性分数只用于发现明显噪声和触发备用引擎，不等于语义相关性证明。

## 输出与验收

每次运行生成：

| 文件 | 内容 |
| --- | --- |
| `output/report.md` | 最终研究报告 |
| `output/plan.json` | 当前显式计划、子问题状态、结构覆盖度与预算 |
| `output/events.jsonl` | 计划创建、选择、更新、评估与结束事件 |
| `output/trace.json` | 本次主图工具调用，不含网页正文和密钥 |
| `output/sources.json` | 搜索/抓取预算、来源修订、失败来源与当前证据图 snapshot |
| `output/evidence.json` | 被 Evidence 使用的来源、canonical Claims、exact quotes、冲突和完整性错误 |
| `output/run.json` | 模型、拓扑、effort、thread、委派角色、基础 token 用量、证据图计数和验收结果 |
| `output/checkpoints.sqlite` | 可恢复的消息、计划、事件、预算、来源修订与 `C#/E#/X#`；不含整页正文 |

抓取成功不等于合格证据：少于 300 个可见字符的页面标记为 `insufficient_content`；300–499 字符的短页只有在同一 host 已存在至少一条 500 字符以上的完整证据时，才以 `evidence_quality=limited` 入账；500 字符以上为 `full`。URL fragment 会被去重，403/404、超时和安全拒绝会作为失败来源记录。每个 canonical URL 保留全部 `content_revisions`；正文相同的不同 URL 会在最新来源快照中标记 `duplicate_of_source_id`。这个 latest duplicate alias 可随重抓正文变化而重算；历史 plan 的“独立来源”门槛不读取该易变字段，而是按该 plan 的 Evidence 边所绑定的 `source_content_sha256`（来源 revision）去重计算。

研究角色可以通过 `get_source_ledger` 读取 canonical `[S#]`，并用 `get_evidence_graph` 读取 `C#/E#/X#`。single 模式由主 Agent 调用 `record_evidence`；multi 模式只把该工具交给 researcher，parent 只能读取证据图并更新 SQ，不能自行登记 Evidence。调用 `record_evidence(source_id, claim, quote, stance)` 时，`claim` 必须是 12–500 个规范化字符的完整命题，`quote` 必须是 12–800 个字符且严格存在于当前进程缓存的规范化网页正文中；新建 Claim 时省略 `claim_id`，代码分配编号。代码保存 quote/hash 和来源修订 hash，但 `supports` 或 `contradicts` 是模型对边的标注，不是自动事实证明。同一 Claim 同时出现支持与反驳 Evidence 时，代码建立一个 `unresolved` 的 `[X#]`，但不自动裁决哪一方正确。

schema v1 还执行四组运行时硬门槛：使整个 plan 完成的 covered 更新必须满足计划所需 `successful_searches` 和 effort 的独立来源数，失败但已消耗预算的搜索尝试不计入成功搜索门槛；multi 模式每个 SQ 的有效委派必须位于当前 `research-step-*` 之后，是 description 以正确 `[SQ:<active-id>]` 开头的 `task(subagent_type="researcher")`，并已收到与 tool-call ID 匹配的成功 `ToolMessage`，未返回、失败、错误 SQ 或旧步骤调用均不算；任一 active SQ 尝试 blocked 时，只要搜索或独立来源 policy gap 仍可恢复且该 SQ 对应的保留预算尚未耗尽，就拒绝 premature blocked；CLI 会重新计算整个 Claim-Evidence-Source/Conflict 闭包。

最终报告只允许四个 H2，顺序固定为 `Short Answer`、`Key Findings`、`Conflicts and Caveats`、`Sources`；H1 或任何额外 heading 都会被拒绝。每条可报告事实必须逐字复制一个 `claim.text`，严格写成 `- <exact claim.text> [C#][S#]`；contested Claim 只能进入冲突节并同时列出支持与反驳来源，contradicted-only Claim 不能作为事实。canonical/cited `[S#]` 只从含一个 `[C#]` 的受约束事实行提取，标题或 Sources 自报的编号不能补足引用门槛。程序据此重写 deterministic Sources；该节每个非空行都必须精确为 `- [S#] <canonical title> — <canonical URL>`，无 bullet 的事实、额外 prose、未知或错配编号都会被拒绝。

## 离线回归

```bash
.venv/bin/python -m unittest discover -s tests -v
```

当前 77 项全量离线测试全部通过，覆盖原有拓扑、策略、搜索、抓取、预算、checkpoint 与来源映射能力，并新增：exact quote 严格子串拒绝、Claim/Evidence/Conflict 编号与闭包、相反 stance 冲突、同 URL 多次内容修订、旧 Evidence 不被新正文覆盖、修订级证据质量、恢复后必须重抓才能新增 Evidence、编号缺口恢复、图篡改检测、schema v0 迁移、schema v1 covered Claim 门槛、最终成功搜索/独立来源门槛、双 provider 异常不计成功搜索、multi 每 SQ researcher 委派、premature blocked 拒绝，以及四节报告、禁止额外 heading、exact claim.text、`[C#][S#]`、严格 Sources 行和 deterministic Sources 校验。

`run.json.api_usage` 汇总当前 plan checkpoint 中 AI 消息的 provider-reported input/output/cache-read token。结构化 planner 调用尚未进入消息状态，multi 模式的嵌套 subagent 用量也可能不进入外层消息；供应商没有返回账单或价目表，所以当前明确记录 `planner_call_included=false` 和 `estimated_cost_usd=null`，不会用猜测价格冒充真实费用。

## 安全边界

`fetch_url` 只允许公网 HTTP(S)，拒绝本机与内网地址，逐跳校验重定向，并限制重定向次数、下载字节数和返回字符数。网页正文和 exact quote 都被当作不可信数据；quote/hash 只能证明“本轮登记的字符串来自本轮抓取并且未被静默改写”，不能证明发布者身份、页面真实性、时效性或 claim 的语义蕴含关系。

这仍然不是浏览器级安全沙箱。生产系统还需要更强的网络隔离、DNS rebinding 防护、域名策略、提示注入检测、人工审批、全局费用控制和可观测性。
