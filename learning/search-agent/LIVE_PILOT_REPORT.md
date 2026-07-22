# TongAgent Controlled Live Pilot — Terminal Report

```yaml
terminal_status: blocked
starting_commit: a474a34bdfdf5686579d2bf1795c767633ee56a1
ending_commit: d2842fcba3cf9a78bcbfd38fe100695f4c8be8f5
branch: codex/benchmark-readiness
push_status: succeeded
pushed_through_commit: d2842fcba3cf9a78bcbfd38fe100695f4c8be8f5
dry_run_status: passed
actual_agent_runs: 0
second_round: false
cost_control: partial
live_artifacts: null
frames_five_task_status: prepared_not_executed
```

本报告是 `TongAgent Controlled Live Pilot` 一次性目标的永久终态。由于执行前
凭证 gate 未通过，本次没有调用真实模型、真实搜索或真实 fetch，也没有生成
任何 live RunResult。按照目标规则，不自动重试或恢复本目标。

报告自身位于其后的 self-containing 文档提交；`ending_commit` 表示完成配置、
数据、schema、测试和正式 pilot 计划的最后一个可自引用提交。

## 1. 分支备份

- Origin：<https://github.com/cantwakeup/TongAgent4pre>
- Branch：
  <https://github.com/cantwakeup/TongAgent4pre/tree/codex/benchmark-readiness>
- 首次非强制 push：成功，备份原五个 Benchmark Ready commits。
- 数据提交非强制 push：成功，远端已包含 `d2842fc`。
- 未使用 `--force`，未修改或合并 `main`，未创建 PR，未删除远端分支。
- 原五个提交未 amend、rebase、reset、拆分或重写。

## 2. Live preflight 与终止原因

### Dry-run

命令：

```bash
cd /home/huiwei/sy/TongAgent/langchain-deepagents/learning/search-agent

.venv/bin/python -m evaluation.cli run \
  --systems simple_react vanilla_deepagents tongagent \
  --dataset evaluation/datasets/live_pilot.example.jsonl \
  --limit 1 \
  --seed 17 \
  --output /tmp/tongagent-controlled-live-pilot \
  --experiment controlled-live-pilot-dry-run \
  --config evaluation/configs/live_pilot.example.json \
  --dry-run
```

结果：

- dataset digest：
  `sha256:9181d9e5e7ac58e20f0c606133ad404dc43f6cddaa7363f3b5cf2eb4a62153ff`
- 3 scheduled / 0 executed / 0 skipped；
- systems：`simple_react`、`vanilla_deepagents`、`tongagent`；
- task：`pilot-python-313`；
- dry-run 没有创建 attempt 或 experiment 输出目录；
- 公平性 fingerprint：
  `sha256:4931bcc09592a15e919842f14f583965665b97682d8cecaad292c408fbe0b4a5`。

### Blocking gate

当前执行环境的 `OPENAI_API_KEY` 缺失或为空。检查只判断是否存在，没有读取、
打印或保存值。目标明确规定缺少必要凭证时不得执行 live pilot，因此：

- 首轮真实 Agent runs：0；
- 第二轮：未发生；
- 没有尝试 `.env`、对话历史密钥、第三方临时密钥或其他隐式凭证；
- 没有为绕过 gate 更换 provider、模型或 endpoint；
- live 终态：`blocked`。

## 3. 模型、provider 与公平边界

三个系统的 resolved preflight 配置共同使用：

| Field | Value |
|---|---|
| Model provider | `openai` |
| Model | `gpt-5.4-nano` |
| Temperature | `0.0` |
| Max output tokens/call | `5,000` |
| Credential allowlist | `OPENAI_API_KEY`（仅变量名） |
| Search identity | `tongagent-web-search` |
| Fetch identity | `tongagent-fetch-url` |
| Judge | `null` |

实际 live tool runtime 不把 backend identity 字符串当作 provider registry：

- search 固定调用生产 `web_search`；
- 主搜索为 DuckDuckGo HTML；
- 低相关或失败时回退 Bing RSS；
- fetch 固定调用生产 `fetch_url`，使用 `httpx` 抓取公开 HTTP(S) 页面。

B1/B2/B3 使用同一主模型和同一 search/fetch 实现。B2 parent、summarizer 和
general-purpose subagent 复用同一模型；B3 planner、main、reviewer/worker
也复用该模型，planner 调用进入共同 model/token 预算。

每 system/task 使用独立进程、状态、attempt 路径和 RunResult。配置在创建
attempt 前要求唯一 fairness fingerprint，并把 resolved config、完整 config
fingerprint 和 fairness fingerprint 写入标准产物。

### 每 run 共同预算

| Limit | Value |
|---|---:|
| Search calls | 4 |
| Fetch calls | 6 |
| Total admitted tool calls | 12 |
| Model calls | 12 |
| Accounted tokens | 100,000 |
| Wall time | 120 seconds |
| Results/search | 5 |
| Page characters | 12,000 |
| Recursion limit | 125 |

TongAgent 没有额外预算。

## 4. Live run table

由于 preflight gate 阻止真实调用，所有运行值都必须保持 N/A 或 `null`，不能
伪造 0 usage 或假 artifact。

| System | Status | Failure gate | Runs | Tool/search/fetch | Token usage | Cost | Wall time | Artifact |
|---|---|---|---:|---|---|---|---|---|
| `simple_react` | not run | `preflight_missing_credential` | 0 | N/A | `null` | `null` | N/A | `null` |
| `vanilla_deepagents` | not run | `preflight_missing_credential` | 0 | N/A | `null` | `null` | N/A | `null` |
| `tongagent` | not run | `preflight_missing_credential` | 0 | N/A | `null` | `null` | N/A | `null` |

`preflight_missing_credential` 是本次目标级 gate 分类，不是伪造的 RunResult
`failure_type`。因为 worker 从未启动，所以没有 terminal failure record。

### Usage 与 cost

- 当前 runner 对缺失 provider usage 保持 `token_usage=null`，不会补 0。
- `estimated_cost` 当前固定为 `null`，没有 price registry 或跨系统美元账本。
- runner 不能可靠执行整个目标的 `$1.00` 美元硬停止，因此
  `cost_control: partial`。
- OpenAI 2026-07-19 官方标准价格列出 `gpt-5.4-nano` 每百万 input tokens
  `$0.20`、cached input `$0.02`、output `$1.25`：
  <https://developers.openai.com/api/docs/pricing>。
- 单题计划的 3 × 100,000 accounted-token ceilings 即使全部按更贵的 output
  rate 计算也是 `$0.375`，低于目标上限；这只是配置上界，不是账单或自动
  hard stop。

## 5. Artifact、引用与安全验收

真实执行没有发生，因此：

- live artifact path：`null`；
- live RunResult：0；
- search/fetch trace：未生成；
- Claim–Evidence–Source：未生成；
- citation/exact quote 验证：未执行；
- provider、限流、网络或解析 failure taxonomy：没有真实样本可验证；
- 没有把单题结果解释为性能结论。

预执行安全审计确认：

- resolved config 只保存凭证环境变量名，不保存值；
- worker 清除隐式 model/search/tracing/endpoint 凭证，仅恢复 allowlist；
- B3 通过 runtime dependency injection，不读取生产 `.env` 初始化路径；
- canonical trace 对常见 token、Bearer、query secret 和大正文做脱敏；
- production/docs/data secret scan 没有发现高置信 credential token；
- 两个配置文件只含公开 endpoint 和环境变量名。

仍未解决的安全/可观测性限制：

1. `worker.stdout.log`、`worker.stderr.log` 和 B3 checkpoint 不是统一的
   fail-closed sanitizer 输出，真实运行后仍须按字面凭证做不打印内容的扫描。
2. canonical JSON artifacts 不保存完整 page body；B3 checkpoint 可能保存
   ToolMessage 正文，但当前没有稳定的后验 verifier 自动证明 evidence quote
   出现在保存正文中。
3. trace sanitizer 对 `id_token` 命名的覆盖弱于 config secret validator。
4. 某些 model provider/auth/rate-limit 异常可能在 run-level 退化为
   `runner_error`，真实运行后需核对 taxonomy。
5. all-systems CLI 会顺序连续执行，但不会在系统之间暂停进行累计美元 gate；
   后续 live 应使用同一 experiment ID 分三条单系统命令，并逐次人工验收。

本次未做代码修复，也没有创建
`fix(eval): harden controlled live pilot` 提交，因为没有真实运行暴露可复现的
工程阻塞；只收紧了受控配置并增加纯离线 preflight 回归。

## 6. FRAMES 5题准备状态

状态：`prepared_not_executed`。

官方来源：

- Dataset：`google/frames-benchmark`
- Config/split：`default` / `test`
- Revision：`58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef`
- Download date：2026-07-19
- License：Apache-2.0
- Source URL：
  <https://huggingface.co/datasets/google/frames-benchmark/resolve/58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef/test.tsv>
- Source SHA-256：
  `4255093c93b595b5b04c7c8dde290b48ec87d72ca0fb0b760d9dd02740d669ff`
- Source size/rows：484,887 bytes / 824 rows
- Eligibility：824 accepted / 0 rejected
- Selection：全量 eligibility 后，
  `random.Random(17).shuffle`，取前 5
- Output SHA-256：
  `e81379800805857fc04599c94f6174566e639d82eb0ccbd6b295ce72f328ee52`

| Task ID | Source index | Reference answer |
|---|---:|---|
| `frames-test-0664` | 664 | `4` |
| `frames-test-0191` | 191 | `Chatsworth House` |
| `frames-test-0123` | 123 | `Halifax` |
| `frames-test-0016` | 16 | `28` |
| `frames-test-0718` | 718 | `85` |

Artifacts：

- `evaluation/datasets/frames_pilot_seed17.jsonl`
- `evaluation/datasets/frames_pilot_seed17.manifest.json`
- `evaluation/prepare_frames_pilot.py`
- `evaluation/configs/frames_pilot_seed17.example.json`
- `FORMAL_PILOT_PLAN.md`

每个 JSONL task 都保存：

- `id`
- `question`
- `reference_answer`
- `source_dataset`
- `source_split`
- `source_index`
- `metadata`

没有人工按难度、答案或系统表现挑题，也没有执行这 5 题。

## 7. 修改与 commit

起始五个 Benchmark Ready commits 保持原样：

1. `c0f559ee4cd2c1e3ea8f312564200aec611e667d`
2. `a9bdeb6bae2e4c7776659ea6e4a2720ca94e1d68`
3. `6d51f71715fca01fdb60ac749395953dec2da267`
4. `85a49b04b0f115daafb3edb50f25f5851928ef31`
5. `a474a34bdfdf5686579d2bf1795c767633ee56a1`

本目标新增数据提交：

- `d2842fcba3cf9a78bcbfd38fe100695f4c8be8f5`
  `data(eval): prepare deterministic FRAMES pilot`

该提交包含：

- 将单题 live 总工具/model 调用上限从 30 收紧为 12；
- 新增 5题 FRAMES 受控配置；
- `EvalTask` 增加可选、all-or-none 的外部 provenance 字段；
- 官方 source hash/size/row count fail-closed 选择器；
- 5题 JSONL、manifest 和复现计划；
- live config/B3 option/schema/selection/hash 离线回归测试；
- benchmark schema 与 FRAMES 接入文档更新。

未修改 Agent 架构、Evidence Graph、Adaptive Control、prompt 或
`search_agent.py`。

## 8. 测试结果

所有测试禁用网络 socket；没有调用模型或搜索 provider。

```bash
.venv/bin/python -m pytest -q \
  --disable-socket --allow-unix-socket \
  tests/test_evaluation_*.py
```

结果：`88 passed, 0 failed, 0 skipped`。

```bash
.venv/bin/python -m pytest -q \
  --disable-socket --allow-unix-socket tests
```

结果：`237 passed, 24 subtests passed, 0 failed, 0 skipped`。

```bash
UV_OFFLINE=1 \
UV_BIN=/home/huiwei/miniconda3/envs/wzq_base/bin/uv \
bash scripts/check_tongagent.sh
```

结果：

- locked sync：68 packages resolved / 66 checked，offline；
- format：50 files already formatted；
- lint：passed；
- compile/import：passed；
- metric regressions：`82 passed, 21 subtests passed`；
- complete offline：`237 passed, 24 subtests passed`；
- gate：`TongAgent checks passed`。

其他：

- controlled config + FRAMES + contract 专项：`19 passed`；
- `git diff --check`：passed；
- high-confidence production/docs/data credential scan：0 matches。

## 9. 下一步正式 5题 pilot

本次目标不会执行以下命令。下一目标开始前必须重新确认：

- `OPENAI_API_KEY` 来自当前安全环境；
- 官方模型价格和账户费用授权；
- branch/HEAD、工作区和 config；
- raw logs/checkpoint 的 post-run secret scan；
- 每系统完成后人工核对 usage、failure、artifact 和累计成本。

先 dry-run：

```bash
cd /home/huiwei/sy/TongAgent/langchain-deepagents/learning/search-agent

EXPERIMENT_ID="frames-pilot-seed17-$(date +%Y%m%d-%H%M%S)"

test -n "${OPENAI_API_KEY:-}" &&
.venv/bin/python -m evaluation.cli run \
  --systems simple_react vanilla_deepagents tongagent \
  --dataset evaluation/datasets/frames_pilot_seed17.jsonl \
  --limit 5 --seed 17 \
  --output output/evaluations \
  --experiment "$EXPERIMENT_ID" \
  --config evaluation/configs/frames_pilot_seed17.example.json \
  --dry-run
```

dry-run 通过后，固定同一 `EXPERIMENT_ID`，逐条执行；每条完成后暂停检查：

```bash
.venv/bin/python -m evaluation.cli run \
  --systems simple_react \
  --dataset evaluation/datasets/frames_pilot_seed17.jsonl \
  --limit 5 --seed 17 \
  --output output/evaluations --experiment "$EXPERIMENT_ID" \
  --config evaluation/configs/frames_pilot_seed17.example.json

.venv/bin/python -m evaluation.cli run \
  --systems vanilla_deepagents \
  --dataset evaluation/datasets/frames_pilot_seed17.jsonl \
  --limit 5 --seed 17 \
  --output output/evaluations --experiment "$EXPERIMENT_ID" \
  --config evaluation/configs/frames_pilot_seed17.example.json

.venv/bin/python -m evaluation.cli run \
  --systems tongagent \
  --dataset evaluation/datasets/frames_pilot_seed17.jsonl \
  --limit 5 --seed 17 \
  --output output/evaluations --experiment "$EXPERIMENT_ID" \
  --config evaluation/configs/frames_pilot_seed17.example.json
```

不得使用 `--rerun`、`--no-resume`、循环自动重试或外部 LLM judge。5 题仍只是
pipeline pilot，不是最终统计结论。

## 10. 永久结束

本目标终态为 `blocked`，原因是首次 live preflight 缺少必要凭证。FRAMES 数据
准备、离线测试、报告和分支备份已完成。不得自动重试 live、重新审计本目标、
下载数据或恢复执行；后续只能由用户明确创建一个新目标。
