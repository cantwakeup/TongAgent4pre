# Stage 03C：正文片段级 Evidence Graph 与严格报告映射

日期：2026-07-17

状态：已完成，可复现

前一阶段：`stage-03b` 搜索质量与证据控制强化

## 一句话成果

TongAgent 不再把“抓取过某个 URL”直接等同于“该页面支持报告事实”：模型必须为每个可报告命题提交 exact quote，代码在当前进程缓存的规范化网页正文中做严格子串校验，再建立 `Claim [C#] -> Evidence [E#] -> Source [S#]` 图；相反证据形成显式 `Conflict [X#]`，最终报告必须逐字使用 canonical `claim.text [C#][S#]`。

## 1. 本阶段解决的问题

Stage 03B 已经保证 `[S#]`、标题和 URL 不会被模型局部重编号，但来源仍然只到整页粒度。BIGAI 真实运行暴露出更深一层的边界：

- researcher 可以从整页正文自由总结，parent 和最终报告无法确定具体用了哪段原文。
- 一个真实 `[S#]` 可以被挂到与正文无关或范围过宽的自然语言 note 上。
- 同一 URL 的正文变化会让“这条 Evidence 当时依据哪个页面版本”变得不清楚。
- 来源之间的支持与反驳只有散文描述，没有机器可读的冲突对象。
- 即使 Sources 列表正确，报告正文仍可能改写 Claim、交叉绑定 `[C#]` 与 `[S#]`，或在没有 Claim 的情况下引用来源。
- 模型可能在搜索或独立来源门槛尚可补足时过早把当前 active SQ 标成 blocked。

Stage 03C 将这些关系变成代码拥有、可恢复、可独立验收的结构。它解决的是 provenance 和结构完整性，不宣称已经自动完成语义事实核查。

## 2. 已完成能力

| 能力 | 代码行为 | 验证证据 |
| --- | --- | --- |
| exact quote 登记 | 模型提交 `claim` 与 `quote`；代码规范化空白后要求 quote 严格存在于本轮抓取正文 | 伪造 quote、过短/过长 quote 回归 |
| Canonical Claim | 代码分配 `C#`；新建时禁止模型自造 ID，复用时要求同 SQ、同 claim.text | 未知 ID、跨 SQ、改写 Claim 回归 |
| Evidence Edge | 代码分配 `E#`，保存 stance、quote/hash、来源修订 hash、URL、标题和质量 | 图完整性与篡改测试 |
| 显式冲突 | 同一 Claim 同时存在 supports/contradicts 时生成一个 `unresolved X#` | 冲突建立、去重和闭包测试 |
| 内容修订 | 同 URL 的每个新正文 hash 追加到 `content_revisions`，旧 Evidence 不被覆盖 | 两次、三次内容变化测试 |
| 独立来源 | 按 plan Evidence 绑定的 `source_content_sha256`/revision 去重；latest duplicate alias 不改写历史门槛 | duplicate、内容漂移与最终门槛测试 |
| Claim-backed SQ | schema v1 covered 必须有 supported/contested Claim，来源集合必须等于 Claim 边的来源集合 | 状态工具与最终闭包测试 |
| 最终计划门槛 | 使 plan 全部 covered 的更新执行最低成功搜索数和 effort 独立来源数检查 | policy gate 回归 |
| multi 每 SQ 委派 | parent 无 `record_evidence`；只接受当前 research step 后、SQ marker 正确且已有匹配成功结果的 `task(researcher)` | 多 Agent 工具隔离与状态工具测试 |
| 防 premature blocked | 任一 active SQ 的 policy 缺口仍能用该 SQ 保留预算补足时拒绝 blocked | 可恢复预算测试 |
| 严格报告 | 仅四个 H2、每行一个 exact claim.text、`[C#][S#]` 闭包，以及 exact canonical Sources；拒绝 H1、额外 heading 和 Sources 伪装事实 | 报告映射与交叉编号测试 |
| 审计产物 | 新增 `evidence.json`，保存参与证据图的来源、C/E/X 和 integrity errors | artifact 与真实运行 |

## 3. 关键调用链

```text
web_search
  |
  v
fetch_url
  |
  +-- 质量门槛 full / limited / insufficient
  +-- canonical Source [S#]
  +-- content_revisions: hash / title / chars / quality
  +-- 规范化正文只进入当前进程内 page cache
  |
  v
single 主 Agent 或 multi researcher 调用 record_evidence(source_id, claim, exact quote, stance)
  |
  +-- quote 长度 12..800
  +-- 严格子串检查
  +-- claim 长度 12..500，必须是可报告命题
  +-- 代码分配 Claim [C#] / Evidence [E#]
  +-- supports + contradicts -> unresolved Conflict [X#]
  |
  v
get_evidence_graph
  |
  v
update_subquestion
  +-- 当前 SQ / 当前 researcher step
  +-- Claim-Evidence-Source/Conflict 闭包
  +-- 最低 `successful_searches` 与 Evidence revision 独立来源
  +-- premature blocked 防护
  |
  v
FINAL SYNTHESIS
  +-- exact claim.text [C#][S#]
  +-- deterministic Sources
  |
  v
report/evidence/plan/events/sources/run + SQLite checkpoint
```

这里没有持久化的 Passage 或 Document 正文节点。模型负责从 `fetch_url` 返回的正文中选择 quote；代码负责确认该 quote 确实是当前缓存正文的严格子串，并固定它所对应的来源修订。single 模式由主 Agent 完成登记；multi 模式只给 researcher 网络工具与 `record_evidence`，parent 只能读取 `get_evidence_graph` 并据此调用 `update_subquestion`，不能自行补造 Evidence。

## 4. Evidence Graph v1

### Source 与内容修订

`[S#]` 仍由 canonical URL 确定。每条成功来源保存首次抓取字段和最新抓取字段，同时保留全部 `content_revisions`：

```text
content_sha256 / latest_content_sha256
title / latest_title
content_chars / latest_content_chars
evidence_quality / latest_evidence_quality
quality_reason / latest_quality_reason
content_changed
content_revisions[]
duplicate_of_source_id
```

每个 revision 至少保存正文 hash、标题、字符数、`full|limited` 和质量原因。相同 URL 后续返回新正文时，只追加 revision 并更新 latest 字段，不改写第一次抓取的元数据。Evidence 自己保存 `source_content_sha256`，因此旧边仍能指向其创建时的 revision。

正文相同的不同 URL 会保留各自 `[S#]`，最新来源快照可把后出现的记录标记为 `duplicate_of_source_id`。该 alias 描述 URL 的 latest fetch，重抓发生内容漂移时可以重算；它不参与历史 Evidence 的 policy 判定。最低来源门槛从当前 plan 的 Evidence 边读取不可变的 `source_content_sha256`，按实际引用的 revision hash 去重，而不是简单统计 URL，或回看来源当前的 duplicate alias。

### Claim `[C#]`

Claim 是模型提出的、自包含、可直接放入报告的命题，保存：

```text
claim_id / subquestion_id / text / status
supporting_evidence_ids / contradicting_evidence_ids / source_ids
```

状态由边集合确定：

- 只有 supports：`supported`
- 只有 contradicts：`contradicted`
- 两者都有：`contested`

新建 Claim 时必须省略 `claim_id`，由代码分配下一个 `C#`。只有在给已成功登记的 Claim 增加另一条 Evidence 时才传回已有 ID；工具拒绝未知 ID、跨 SQ ID 和不同的 canonical claim.text。

### Evidence `[E#]`

每条 Evidence 保存：

```text
evidence_id / claim_id / subquestion_id / source_id
stance / quote / quote_sha256 / source_content_sha256
url / title / evidence_quality
```

`quote` 会先折叠空白，再与当前进程内该 `[S#]` 的规范化正文做严格子串检查。代码保证“保存的 quote 来自刚抓到的页面正文”，但不会自动判断 quote 是否在语义上蕴含 claim；`supports` 或 `contradicts` 是模型提交的 stance。

同一 Claim、来源、stance 和 quote 会复用已有 `E#`。同一来源的同一 quote 不能既支持又反驳同一 Claim。

### Conflict `[X#]`

当同一 Claim 同时有 supports 与 contradicts Evidence 时，代码创建一个 `status=unresolved` 的 Conflict，并同步两侧 Evidence、来源和 SQ。Conflict 只是机器可读的未决关系：系统不会根据来源数量、发布时间或机构权威性自动裁决哪一侧正确。

## 5. exact quote 的可信边界

Stage 03C 采用的是“模型选片段、代码验原文”方案：

1. `fetch_url` 返回规范化前的可见正文给模型，并把规范化版本缓存于进程内。
2. 模型填写一个可报告 `claim`，另行复制 12–800 字符 exact quote。
3. `record_evidence` 折叠 quote 的空白，要求它严格出现在缓存正文中。
4. 成功后代码保存 canonical quote、quote hash 和来源正文 hash。
5. 最终报告复制的是 `claim.text`，不是 quote；quote 留在 `evidence.json` 供审计。

因此，模型不能登记页面中不存在的 excerpt；但它仍可能把一个真实 quote 标成并不恰当的 supports/contradicts，或写出范围大于 quote 的 Claim。当前 reviewer 不是所有 effort 的硬门槛，也不是代码级语义判定器：只有 high/xhigh multi 按既有策略要求 reviewer，low/medium 的 Claim stance 仍可直接来自研究模型。

## 6. schema v1 状态机硬门槛

新计划带 `evidence_schema_version=1`。每个 SQ 额外保存 `claim_ids` 和 `conflict_ids`。

`update_subquestion(status=covered)` 必须满足：

1. 只能更新当前 active SQ。
2. multi 模式已有位于当前 `research-step-*` 之后的 `task(subagent_type="researcher")`；其 description 以正确 `[SQ:<active-id>]` 开头，并存在 tool-call ID 匹配、`status=success` 的 `ToolMessage`。未返回、失败、错误 SQ marker 或旧 research step 的调用均不算。
3. 当前 SQ 至少有一个 `supported` 或 `contested` canonical Claim。
4. 提交的 `evidence_source_ids` 必须与这些 Claim 的 Evidence 来源集合完全相同。
5. Claim 不能跨 SQ；Conflict 必须与 contested Claim 闭包一致。
6. 如果该更新会完成整个 plan，`successful_searches` 必须达到 `min(max_searches, SQ 数)`；失败但消耗预算的搜索尝试只进入调用/额度计数，不算成功搜索。尤其是 DuckDuckGo 与 Bing provider 都异常时，工具返回 `status=error`，不能增加 `successful_searches`。
7. 如果该更新会完成整个 plan，plan 必须引用 effort 所需数量的独立成功来源；计数按这些 Claim 的 Evidence 边所绑定的 `source_content_sha256`/revision hash 去重，不读取 URL 当前的 `duplicate_of_source_id`。

任一 active SQ 若尝试 blocked，代码都会检查搜索/独立来源 policy gap 是否仍可恢复，以及该 SQ 对应的搜索/抓取保留预算是否尚未耗尽。只要缺口仍有对应的可恢复空间，就拒绝 premature blocked，让模型继续 `web_search/fetch_url/record_evidence`，或交给外层有界 retry。对应预算耗尽，或已经无法形成 supported Claim 后，仍可诚实 blocked。

Stage 03B 旧 plan 没有 schema 字段时迁移为 v0，并补空 `claim_ids/conflict_ids`，不会伪造历史 Claim；v1 的新硬门槛只用于新计划。

## 7. checkpoint 语义

外层 checkpoint 保存：

- 来源目录、`next_source_sequence`、全部 revision hash/标题/字符数/质量元数据；
- `C#/E#/X#`、canonical claim.text、exact quote、quote hash 和来源 revision hash；
- plan、SQ 绑定、事件和每 SQ 预算。

它明确不保存整页正文。进程恢复后，既有 Evidence 仍可通过其 quote/hash 和 revision 元数据验证图闭包，但进程内 page cache 为空；若要为旧 `[S#]` 登记新的 Evidence，必须重新抓取 canonical URL。重抓相同正文会恢复 cache；重抓到新正文会追加 `content_revisions`，不会让旧 Evidence 静默漂移。

已完成 thread 开始新 plan 时保留 thread-stable `[S#]` 目录和 revision 历史，但清空 plan-local C/E/X 图并重置工具预算。

## 8. 严格报告协议

schema v1 报告只允许四个 H2，顺序固定：

```text
## Short Answer
## Key Findings
## Conflicts and Caveats
## Sources
```

不允许 H1、其他 heading 或额外小节。Short Answer 和 Key Findings 的每个非空事实行只能包含一个 Claim，形状必须是：

```text
- <exact canonical claim.text> [C#][S#]
```

- 不能本地改写、否定、加前后缀或把多个 Claim 合并到一行。
- `[S#]` 必须来自该 Claim 的 Evidence 边。
- `supported` Claim 只能进入 Short Answer 或 Key Findings。
- `contested` Claim 只能进入 Conflicts and Caveats，并同时引用支持与反驳来源。
- `contradicted` Claim 不能作为报告事实。
- 普通过程或证据质量 caveat 可以写在冲突节，但不能冒用 `[C#]` 或 `[S#]`。

模型完成 `/report.md` 后，程序只从受约束事实节中“恰好含一个 `[C#]`”的行提取 cited `[S#]`，再按首次出现顺序重建 Sources。H1、其他 heading、无 Claim 行或模型自己写的 Sources 行都不能贡献 cited ID：

```text
- [S#] <canonical title> — <canonical URL>
```

Sources 节的每个非空行都必须精确符合上述 bullet 形状；无 bullet 的事实、URL prose 或额外 heading 不会被 canonicalizer 吞掉，而会导致验收失败。随后 CLI 再检查四节结构、Claim 是否全部出现、claim.text 是否逐字一致、C/S 是否同边、冲突是否两侧完整、Sources 是否 canonical。该协议大幅缩小报告自由改写空间，但仍不能识别所有没有进入受约束事实行的隐性语义暗示。

## 9. 审计产物

Stage 03C 新增：

| 文件 | 内容 |
| --- | --- |
| `output/evidence.json` | graph version、被 Evidence 使用的来源修订、Claims、Evidence units、Conflicts 和 integrity errors |

现有文件同时扩展：

- `sources.json` 保存完整预算、来源 revision 目录和 C/E/X snapshot。
- `plan.json` 的每个 SQ 保存 `claim_ids/conflict_ids`。
- `run.json.research` 保存 Claim 状态计数、Evidence/Conflict 数量、integrity error 数和 evidence 路径。
- `run.json.validation` 保存来源节是否被 deterministic canonicalizer 重写以及图完整性错误。

`trace.json` 继续只记录工具名、参数摘要和结果大小，不保存网页正文。`evidence.json` 会保存经过登记的短 quote，因此它与 trace 的数据边界不同。

## 10. 离线验证

完整命令：

```bash
.venv/bin/python -m unittest discover -s tests -v

ruff check \
  research_state.py research_graph.py evidence_graph.py \
  telemetry.py search_agent.py agent_policy.py tests

ruff format --check \
  research_state.py research_graph.py evidence_graph.py \
  telemetry.py search_agent.py agent_policy.py tests
```

结果：

当前 77 项全量离线测试全部通过，Ruff check 与格式检查也通过。

Stage 03C 新增或强化的覆盖包括：

- 伪造、过短、过长和规范化后不存在的 quote 拒绝；
- Claim 不是短标签，未知/跨 SQ/改写 C# 拒绝；
- C/E/X 稳定编号、重复边去重和相反 stance 冲突；
- 同 URL 两次、三次正文修订与旧边完整性；
- revision 级 `full/limited` 质量不被最新页面覆盖；
- checkpoint 恢复后新增 Evidence 必须先重抓 URL；
- 删除中间 ID 后继续使用最大序号，不发生碰撞；
- quote hash、stance、revision、闭包篡改检测；
- schema v0 plan 迁移不伪造 Claim；
- schema v1 covered、最终 `successful_searches`、Evidence revision 独立来源数和任一 active SQ premature blocked 门槛；
- DuckDuckGo/Bing provider 都异常时消耗预算但不增加 `successful_searches`；
- multi 当前 SQ researcher 委派的 research-step、SQ marker、tool-call ID 与成功 ToolMessage 闭包；
- multi parent 无 `record_evidence`，只有 researcher 能登记新 Evidence；
- 四个 H2、禁止 H1/额外 heading、exact claim.text、单 Claim 行、C/S 配对、冲突位置、仅从含 C 的事实行提取 cited ID，以及 exact canonical Sources。

## 11. 真实路径验证

### 11.1 无模型 BIGAI full + limited smoke

直接调用真实抓取与预算工具层，先抓取 `https://www.bigai.ai/about/`，再抓取 `https://www.bigai.ai/tongprogram-2026/`：

```text
about/             1488 visible chars  full
tongprogram-2026/   430 visible chars  limited
```

后者只有在同 host 已有 full 锚点后才入账。该 smoke 不调用模型，验证真实 HTML 提取、来源 hash/revision、同 host 短页门槛和 exact-quote 可用正文路径。

### 11.2 low / single 通过

复现模板：

```bash
.venv/bin/python search_agent.py \
  --effort low \
  --mode single \
  --model gpt-5.4-nano \
  --thread-id stage03c-low-pass-your-unique-id \
  --no-stream \
  --print-report \
  "在唯一原子子问题内搜索并收集两个独立官网来源：抓取 https://www.bigai.ai/about/ 注册主谓完整的机构介绍 claim；抓取 https://www.bigai.ai/tongprogram-2026/ 注册主谓完整的通计划联系邮箱 claim。新建 claim 省略 claim_id。成功搜索数和两个来源未满足前不得 covered 或 blocked；工具提示不足就继续调用 web_search/fetch_url/record_evidence。不得用 info.bigai@bigai.ai 代替 tongprogram@bigai.ai。最终逐字复制 claim.text 并标 [C#][S#]；普通 caveat 不带 C/S。"
```

通过记录：

```text
thread:              stage03c-low-pass-final-20260717-022715
plan status:         completed
coverage:            1.0
validation:          passed
searches:            1 / 2
fetches:             3 / 3
independent sources: 2
claims/evidence:     2 / 2
S1:                  about, full
S2:                  TongProgram-2026, limited
```

这条路径证明 single Agent 能在一个原子 SQ 内满足最低独立来源门槛，并把 full 与 limited 页分别登记成 canonical Claim。

### 11.3 窄范围 medium / multi 通过

复现模板：

```bash
.venv/bin/python search_agent.py \
  --effort medium \
  --mode multi \
  --model gpt-5.4-nano \
  --worker-model deepseek-v4-flash \
  --thread-id bigai-stage03c-multi-smoke-your-unique-id \
  --no-stream \
  --print-report \
  "这是 Stage 03C 的窄范围机制验收，只建立两个原子子问题，不研究论文、项目、开源、产业合作或通计划的广泛目标。SQ1 仅核验 BIGAI 官网 about 页所述机构性质、成立背景和院长；SQ2 仅核验 https://www.bigai.ai/tongprogram-2026/ 是否公开了通计划联系邮箱及邮箱值。每个 SQ 必须委派 researcher、至少搜索一次、抓取对应官网页，并把 exact quote 注册为主谓完整 canonical Claim；新建时省略 claim_id。普通 caveat 不得带 C/S。"
```

通过记录：

```text
thread:               stage03c-final-multi-20260717-release-net2
plan status:          completed
coverage:             1.0
validation:           passed
researcher tasks:     2（每个 SQ 一次）
successful searches:  2
fetches:              3 / 6
claims/evidence:      2 / 2
conflicts:            0
TongProgram:          limited
parent record_evidence: 0
graph integrity errs: 0
```

每个 SQ 由一个 researcher task 完成搜索、抓取和 Evidence 登记；SQ1 与 SQ2 各形成一个 canonical Claim/Evidence。parent 没有、也没有调用 `record_evidence`，只读取图并更新 SQ；TongProgram 页保持 `limited`，整条 release 验收最终 `validation=passed`。

### 11.4 完整宽范围 medium / multi：诚实 partial

完整 BIGAI 问题要求同时核验机构、团队、论文、项目、开源、产业合作，以及“通计划”的正式名称、发布主体、目标、参与方式和公开进展。实际记录：

```text
thread:           bigai-stage03c-medium-gated-20260717-023334
plan status:      partial
coverage:         0.5
searches:         4 / 4
fetches:          6 / 6
sources:          3
claims/evidence:  4 / 4（全部属于 SQ1）
SQ1:              covered
SQ2:              blocked
```

TongProgram 页被成功抓取为 `limited`，但其可核验正文只足以支持联系邮箱这一窄事实，不能支持“正式名称、发布主体、目标、参与方式、公开进展”整组要求。系统没有把一个真实但过窄的 short-page 事实扩张成完整 SQ2 Claim，因此最终保持 partial 0.5，CLI 按“plan 未完成”诚实拒绝通过。

这条失败是 Stage 03C 的正向证据：新 Evidence Graph 没有让搜索覆盖面自动变广，而是阻止来源级 `[S#]` 被用来掩盖正文缺口。若要完成宽范围问题，需要更多政府、高校或项目官方页面和更高研究预算。

## 12. 代码索引

- `evidence_graph.py`：文本规范化、exact quote 校验、C/E/X store、revision/图完整性与报告映射校验。
- `research_state.py`：Claim、Evidence、Conflict、schema v1 plan 与 budget snapshot 类型。
- `research_graph.py`：Claim 闭包、最终 policy gate、multi 委派、premature blocked、防错工具与严格报告 prompt。
- `search_agent.py`：来源 revision、进程内正文 cache、工具装配、deterministic Sources、CLI 验收与 `evidence.json`。
- `tests/test_evidence_graph.py`：exact quote、内容修订、冲突、图完整性和报告协议。
- `tests/test_research_graph.py`：schema v1 SQ 状态机、policy gate、multi 委派和恢复。
- `tests/test_research_budget.py`：来源 revision、质量、去重和 checkpoint。
- `tests/test_source_mapping.py`：deterministic Sources 与来源映射。

## 13. 已知限制与下一阶段

- exact quote 校验只证明字符串来自本轮抓取正文；Claim 与 quote 的语义支持关系仍由模型判断。
- `supports/contradicts` 是模型 stance，不是自动 NLI、事实证明或来源权威度评分。
- reviewer 不是 schema v1 的统一语义硬门槛：只有 high/xhigh multi 按既有 effort 策略强制 reviewer，且 reviewer 仍输出自然语言建议。
- Conflict 只有在模型把相反 Evidence 复用到同一个 canonical C# 时才建立；系统不会自动合并语义相同但措辞不同的 Claims。
- `unresolved X#` 不会自动按机构权威性、发布日期或来源数量裁决。
- coverage 是 Claim 闭包强化后的 SQ 结构覆盖率，不代表宽 SQ 的每个需求字段都已满足。
- checkpoint 不保存整页正文；恢复后若要新增 Evidence，必须重抓 URL，并再次消耗该 SQ 的 fetch 预算。
- 内容 hash 是完整性和 revision 标识，不是网页签名、时间戳证明或发布者身份认证。
- 页面正文仍受字符截断、HTML 提取质量、JavaScript 动态内容和 PDF 支持边界影响。
- deterministic Sources 与严格 Claim 行能约束报告形状，但不能理解所有无引用 caveat 中可能暗含的事实主张。
- 输出目录仍是 last-run artifact，不支持同目录并发隔离；token 统计仍不含 planner、部分 nested subagent 和真实账单金额。

下一阶段更适合加入：来源可信度与时效评分、独立语义支持 reviewer、Claim 归一化与近义冲突发现、按证据缺口自适应追加预算，以及宽范围研究的质量/成本对照实验。

## 14. PPT 建议页

1. Stage 03B 为什么只能证明“引用了正确 URL”，不能证明“正文支持事实”。
2. exact quote 方案：模型选原文，代码做严格子串验证。
3. C/E/S/X 图与内容 revision hash。
4. schema v1 四组硬门槛：Claim 闭包、`successful_searches`/Evidence revision 独立来源、当前 step 的成功 SQ researcher 委派、任一 active SQ premature blocked。
5. checkpoint 为什么保留 quote/hash 而不保存整页正文。
6. 严格四节报告与 deterministic Sources。
7. low/single 与窄 medium/multi 两条真实通过轨迹。
8. 宽范围 BIGAI partial 0.5：为什么诚实失败比来源级 coverage 1.0 更有价值。
9. provenance 已加强，但语义事实核查与自适应计算仍是下一阶段。
