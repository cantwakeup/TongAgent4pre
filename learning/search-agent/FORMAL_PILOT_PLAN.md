# FRAMES 5-Task Benchmark Pipeline Pilot

本计划只准备下一阶段的 5 题 × 3 系统 pipeline pilot，不在本次一次性目标中
执行。它不是正式 benchmark 结果，也不能用于系统性能排名。

## 为什么先运行 5 题

5 题足以暴露真实 provider、搜索、抓取、预算、artifact、resume、聚合和
normalized exact match 的集成问题，同时把首次多题实验限制在 15 个 Agent
runs。它不足以估计总体准确率、置信区间或不同题型上的稳定差异。

## 数据来源与确定性选择

- Dataset：`google/frames-benchmark`
- Revision：`58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef`
- Config/split/source file：`default` / `test` / `test.tsv`
- License：Apache-2.0
- Source SHA-256：
  `4255093c93b595b5b04c7c8dde290b48ec87d72ca0fb0b760d9dd02740d669ff`
- Download date：2026-07-19
- Source rows：824
- Eligibility：824 accepted / 0 rejected
- Selection：对全部合格行使用 Python `random.Random(17).shuffle`，取前 5
- Selected source indices：`664, 191, 123, 16, 718`

完整 provenance、eligibility rule、输出 hash 和 task ID 位于
`evaluation/datasets/frames_pilot_seed17.manifest.json`。选择不是基于难度、
答案内容或任何系统结果。

复现数据准备：

```bash
cd /home/huiwei/sy/TongAgent/langchain-deepagents/learning/search-agent

curl -L \
  -o /tmp/frames-benchmark-test.tsv \
  https://huggingface.co/datasets/google/frames-benchmark/resolve/58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef/test.tsv

printf '%s  %s\n' \
  4255093c93b595b5b04c7c8dde290b48ec87d72ca0fb0b760d9dd02740d669ff \
  /tmp/frames-benchmark-test.tsv |
sha256sum --check

.venv/bin/python -m evaluation.prepare_frames_pilot \
  --source /tmp/frames-benchmark-test.tsv \
  --output evaluation/datasets/frames_pilot_seed17.jsonl \
  --manifest evaluation/datasets/frames_pilot_seed17.manifest.json \
  --download-date 2026-07-19 \
  --seed 17 \
  --limit 5
```

## 公平配置

三个系统共同使用：

- model：OpenAI `gpt-5.4-nano`；
- search：生产 `web_search`，实际为 DuckDuckGo HTML，低相关或失败时回退
  Bing RSS；
- fetch：生产 `fetch_url` 公网 HTTP(S) 抓取；
- 每 run：4 search、6 fetch、12 total tools、12 model calls、50,000
  accounted tokens、120 秒；
- 同一 dataset digest、seed=17、judge=`null` 和公平性 fingerprint；
- 每个 system/task 独立 worker、attempt 与 RunResult。

配置文件为 `evaluation/configs/frames_pilot_seed17.example.json`。当前 runner
没有美元级硬停止，`estimated_cost` 也保持 `null`，所以 cost control 仍是
`partial`。按 2026-07-19 OpenAI 官方标准价格，`gpt-5.4-nano` 为每百万 input
tokens `$0.20`、cached input `$0.02`、output `$1.25`。15 runs × 50,000
tokens 即使全部按更贵的 output rate 计算，上界也是 `$0.9375`；真实执行仍须
由操作者确认价格、账户和预算授权。价格来源：
<https://developers.openai.com/api/docs/pricing>。

TongAgent planner、reviewer、worker 与主图使用相同模型并计入共同 model/token
预算，不获得额外调用。TongAgent 专属 `evidence_count` 和
`structural_subquestion_coverage` 只作 B3 内部诊断；B1/B2 保持 `null`，不得
据此扣分。

## 指标

主指标：

- `normalized_exact_match`：prediction/reference 分别进行 NFKC、casefold 和
  空白折叠后做 whole-string equality。

辅助指标：

- completion/failure distribution；
- tool/search/fetch calls；
- wall time；
- provider-reported token usage；
- estimated cost（当前不可用时必须为 `null`）。

5 题只验证 pipeline，不产生可推广的统计结论。

## 精确执行顺序

先确认 `OPENAI_API_KEY` 仅存在于当前安全环境，重新核对官方价格，并运行零调用
dry-run：

```bash
cd /home/huiwei/sy/TongAgent/langchain-deepagents/learning/search-agent

EXPERIMENT_ID="frames-pilot-seed17-$(date +%Y%m%d-%H%M%S)"

test -n "${OPENAI_API_KEY:-}" &&
.venv/bin/python -m evaluation.cli run \
  --systems simple_react vanilla_deepagents tongagent \
  --dataset evaluation/datasets/frames_pilot_seed17.jsonl \
  --limit 5 \
  --seed 17 \
  --output output/evaluations \
  --experiment "$EXPERIMENT_ID" \
  --config evaluation/configs/frames_pilot_seed17.example.json \
  --dry-run
```

随后使用同一 `EXPERIMENT_ID`，按顺序分别执行并在每条命令后检查 artifact、
usage、failure 和累计成本：

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

不得使用 `--rerun`、`--no-resume` 或循环自动重试。

## 扩大到 20–30 题的条件

只有同时满足以下条件才扩大：

1. 15 个 pilot runs 全部产生合法 RunResult 或明确 failure record；
2. search/fetch、usage、failure taxonomy、artifact 和 resume 已人工抽查；
3. 没有凭证、Cookie、Authorization header 或隐私内容落盘；
4. quote/citation provenance 有稳定、可审计的后验验证路径；
5. provider rate limit、并发、缓存、网络和费用方案已预注册；
6. 美元预算能被可靠硬停止，或操作者明确接受分系统人工 gate；
7. 官方 evaluator/normalized EM 报告协议固定；
8. 新样本仍按预先声明的全量 eligibility + seeded selection 选择。

未满足任一条件时，应保留 5 题结果为 pipeline smoke，不扩大规模。
