# TongAgent 真实联网研究 Agent

这是一个建立在 LangChain、LangGraph 和 Deep Agents 之上的可运行研究 Agent。它会先生成显式研究计划，再按子问题真实搜索网页、抓取证据、计算结构覆盖度并生成带来源编号的报告，同时保存计划状态、工具轨迹、来源账本和 checkpoint。

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
  -> 结构 coverage 合格后 write_file 写入报告
  -> 质量门槛验收
  -> report.md + plan.json + events.jsonl + trace/sources/run + SQLite checkpoint
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
| `multi` | 主 Agent 不拥有网络工具，必须通过 `task(researcher)` 收集证据 |
| `auto` | `high/xhigh` 或复杂、较长问题走 multi，其余走 single |

在需要审稿的档位，multi 路径为：

```text
主 Agent -> task(researcher) -> 形成草稿
        -> task(reviewer)   -> 修正问题
        -> write_file       -> 最终报告
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

用 `--checkpoint-db /path/to/research.sqlite` 可指定数据库。恢复线程时，模型能看到旧消息；如果 `checkpoint.next` 表明图仍有待执行节点，CLI 会先用 `invoke(None)` 精确继续旧节点，再接收真正的新问题。预算计数、来源 ID 和事件也会从 checkpoint 恢复；新生成的 `trace.json` 仍只记录本次 CLI 进程新增的消息。

当前恢复保证在外层子问题节点边界生效。内层 Agent 返回时，来源账本会和节点结果一起写入外层状态；如果进程恰好终止在网络工具内部，该工具节点仍可能重跑。未完成 plan 会恢复剩余硬预算；已完成 thread 的新 plan 会保留累计来源目录和连续 `[S#]` 编号，但搜索/抓取计数从零开始，避免新问题继承已经耗尽的旧额度。

`covered` 是结构状态，不等于系统已自动理解证据语义。代码只允许当前子问题更新，并要求 `[S#]` 存在于成功来源账本、至少一个来自当前研究步骤；“页面是否真正支持该子问题”目前仍由模型判断，正文片段和 Claim-Evidence 校验属于下一阶段。

每个 plan 创建后，代码会按 SQ 数量确定性切分搜索和抓取额度。例如 medium 的两个 SQ 各自保留 `2 search / 3 fetch`，SQ1 不能消费 SQ2 的份额；checkpoint 会同时恢复当前 SQ、各 SQ 上限和已经使用的次数。进入最终报告阶段后不再激活任何 SQ，因此网络工具不能继续消耗研究预算。

搜索结果会携带 `engine` 与 `relevance_score`。当 DuckDuckGo 失败、为空，或相关结果少于两条时，代码自动查询 Bing，并在结果中记录 `fallback_reason`、`search_quality` 和 `relevant_results`。相关性分数只用于发现明显噪声和触发备用引擎，不等于语义相关性证明。

## 输出与验收

每次运行生成：

| 文件 | 内容 |
| --- | --- |
| `output/report.md` | 最终研究报告 |
| `output/plan.json` | 当前显式计划、子问题状态、结构覆盖度与预算 |
| `output/events.jsonl` | 计划创建、选择、更新、评估与结束事件 |
| `output/trace.json` | 本次主图工具调用，不含网页正文和密钥 |
| `output/sources.json` | 搜索/抓取预算、成功来源和失败来源 |
| `output/run.json` | 模型、拓扑、effort、thread、委派角色、基础 token 用量和验收结果 |
| `output/checkpoints.sqlite` | 可恢复的消息、计划、事件与预算状态 |

抓取成功不等于合格证据：少于 300 个可见字符的页面标记为 `insufficient_content`；300–499 字符的短页只有在同一 host 已存在至少一条 500 字符以上的完整证据时，才以 `evidence_quality=limited` 入账；500 字符以上为 `full`。URL fragment 会被去重，403/404、超时和安全拒绝会作为失败来源记录。

所有角色都可以通过 `get_source_ledger` 读取 canonical `[S#]`、标题和 URL。`update_subquestion` 拒绝不存在于账本的 ID，并要求挂载的证据至少有一条来自当前研究步骤。最终验收还会拒绝未知 ID、未挂载到当前 plan 的旧来源，以及没有在同一来源行严格绑定 canonical ID、标题、URL 的本地重编号或错配。

## 离线回归

```bash
.venv/bin/python -m unittest discover -s tests -v
```

当前 39 项离线测试覆盖拓扑路由、四档策略、角色构建、工具硬预算、每 SQ 预算切片与 checkpoint 恢复、无 active SQ 时的网络拒绝、来源编号与去重、同域完整证据锚定短页、未锚定短页拒绝、DDG 低相关触发 Bing、双搜索引擎失败降级、中文实体噪声识别、canonical ledger 工具、逐行 ID/标题/URL 绑定、HTTP 403、流式正文隔离、planner fallback、计划历史、依赖级联阻塞、账本来源校验、结构覆盖度、尝试上限、message-only checkpoint 形状兼容、CLI `invoke(None)` 续跑、基础 token 聚合，以及 SQLite 关闭重开后的计划与预算恢复。

`run.json.api_usage` 汇总当前 plan checkpoint 中 AI 消息的 provider-reported input/output/cache-read token。结构化 planner 调用尚未进入消息状态，multi 模式的嵌套 subagent 用量也可能不进入外层消息；供应商没有返回账单或价目表，所以当前明确记录 `planner_call_included=false` 和 `estimated_cost_usd=null`，不会用猜测价格冒充真实费用。

## 安全边界

`fetch_url` 只允许公网 HTTP(S)，拒绝本机与内网地址，逐跳校验重定向，并限制重定向次数、下载字节数和返回字符数。网页正文被当作不可信数据。

这仍然不是浏览器级安全沙箱。生产系统还需要更强的网络隔离、DNS rebinding 防护、域名策略、提示注入检测、人工审批、全局费用控制和可观测性。
