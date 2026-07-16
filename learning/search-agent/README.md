# TongAgent 真实联网研究 Agent

这是一个建立在 LangChain、LangGraph 和 Deep Agents 之上的可运行研究 Agent。它会真实搜索网页、抓取证据、生成带来源编号的报告，并把运行策略、工具轨迹、来源账本和 checkpoint 保存到本地。

## 当前闭环

```text
用户问题
  -> 资源策略（effort）与拓扑路由（mode）
  -> single：主 Agent 直接搜索和读取
     或 multi：主 Agent -> researcher -> 可选 reviewer
  -> 结构化来源账本 [S1] [S2] ...
  -> write_file 写入报告
  -> 质量门槛验收
  -> report.md + trace.json + sources.json + run.json + SQLite checkpoint
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

| effort | 默认主模型 | 搜索/抓取上限 | 最少成功来源 | multi reviewer |
| --- | --- | ---: | ---: | --- |
| `low` | `gpt-5.4-nano` | 2 / 3 | 2 | 否 |
| `medium` | `gpt-5.4-nano` | 4 / 6 | 2 | 否 |
| `high` | `gpt-5.4-mini` | 8 / 9 | 3 | 是 |
| `xhigh` | `gpt-5.4-mini` | 12 / 14 | 4 | 是 |

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

用 `--checkpoint-db /path/to/research.sqlite` 可指定数据库。恢复线程时，模型能看到旧消息，但新生成的 `trace.json` 只记录本次运行事件。

## 输出与验收

每次运行生成：

| 文件 | 内容 |
| --- | --- |
| `output/report.md` | 最终研究报告 |
| `output/trace.json` | 本次主图工具调用，不含网页正文和密钥 |
| `output/sources.json` | 搜索/抓取预算、成功来源和失败来源 |
| `output/run.json` | 模型、拓扑、effort、thread、委派角色和验收结果 |
| `output/checkpoints.sqlite` | 可恢复的 LangGraph 消息状态 |

抓取成功不等于合格证据：少于 500 个可见字符的页面标记为 `insufficient_content`；URL fragment 会被去重；403/404、超时和安全拒绝会作为失败来源记录。主程序还会检查成功来源数、来源引用、`write_file`、researcher/reviewer 委派和 Sources URL。

## 离线回归

```bash
.venv/bin/python -m unittest discover -s tests -v
```

测试覆盖拓扑路由、四档策略、角色构建、工具硬预算、来源编号与去重、短页面拒绝、HTTP 403 恢复、流式正文隔离，以及 SQLite checkpoint 恢复。

## 安全边界

`fetch_url` 只允许公网 HTTP(S)，拒绝本机与内网地址，逐跳校验重定向，并限制重定向次数、下载字节数和返回字符数。网页正文被当作不可信数据。

这仍然不是浏览器级安全沙箱。生产系统还需要更强的网络隔离、DNS rebinding 防护、域名策略、提示注入检测、人工审批、全局费用控制和可观测性。
