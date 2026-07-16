# Stage 01：真实单 Agent 搜索闭环

日期：2026-07-15 至 2026-07-16

状态：已完成，可复现

上游基线：`langchain-ai/deepagents@4ddb361b99857c1fc23afb9ada0a68162c190f74`

## 一句话成果

我们从 Deep Agents 的最小工具循环出发，完成了一个使用真实模型、真实搜索、真实网页读取和真实文件落盘的研究 Agent，并为流式观察、网络失败恢复、工具轨迹审计和最低质量验证补上了工程边界。

## 1. 阶段目标

- 建立 Deep Agents、LangChain、LangGraph 三层关系的主干心智模型。
- 用无网络实验观察一次标准 Agent Loop。
- 对比虚拟文件系统与真实操作系统能力边界。
- 实现一个不依赖 fake model 的真实联网搜索 Agent。
- 保存可复现环境、运行轨迹、测试和中文讲解材料。

## 2. 已完成能力

| 能力 | 实现 | 验证方式 |
| --- | --- | --- |
| 模型决策 | 私有 OpenAI-compatible API，经 `ChatOpenAI` 接入 | 真实工具调用轨迹 |
| 搜索 | DuckDuckGo HTML，Bing RSS 后备 | 多查询真实运行 |
| 网页读取 | 公网 HTTP(S) 下载、HTML 文本提取 | 官方页面与普通网页实测 |
| Agent Loop | 模型调用工具，结果回写 state，模型继续决策 | LangGraph `messages` 与 `values` 流 |
| 报告写入 | `FilesystemBackend` 将 `/report.md` 映射到真实 `output/report.md` | 文件落盘检查 |
| 流式观察 | 模型文本、工具调用、工具结果大小实时显示 | CLI 真实运行 |
| 错误恢复 | 403/404、连接失败和安全拒绝转换为工具结果 | 403 单元测试与 Aqours 复测 |
| 安全边界 | 拒绝内网地址，逐跳校验重定向，限制下载与返回字符数 | 源码检查与拒绝测试 |
| 审计 | 保存不含密钥和大段正文的 `trace.json` | 运行后轨迹检查 |
| 最低质量门槛 | 检查搜索次数、读取次数、`write_file` 和 Sources URL | 主程序结束校验 |

## 3. 架构

```text
用户问题
  -> Deep Agents harness
     -> LangChain 模型与工具抽象
     -> LangGraph messages state / Agent Loop
  -> 模型决定下一步
     -> write_todos
     -> web_search
     -> fetch_url
     -> write_file
  -> 工具结果作为 ToolMessage 回到 state
  -> 模型继续搜索、阅读或完成回答
  -> output/report.md + output/trace.json
```

三层职责：

- LangChain：统一模型、工具和消息接口。
- LangGraph：运行状态图、维护 messages、执行模型与工具循环、提供 streaming。
- Deep Agents：在前两者之上组装 todo、filesystem、subagent、summarization 等完整 harness。

我们自己的业务层负责搜索、网页读取、安全策略、研究协议、报告格式和验收规则。

## 4. 关键设计决定

### 使用 uv 独立环境

搜索 Agent 有独立的 `pyproject.toml`、`uv.lock` 和 `.venv`。这符合上游仓库规范，也避免 Conda、pip 与 uv 混用。

### 文件路径是虚拟映射，结果是真实落盘

Agent 看到 `/report.md`，`FilesystemBackend` 把它限制并映射到 `output/report.md`。这不是 fake 文件，而是受根目录约束的真实文件写入。

### 不授予 shell

搜索任务只需要网络读取和报告写入，因此使用 `FilesystemBackend`，不提供任意命令执行能力，减少提示注入和误操作影响面。

### 网络失败是观察结果，不是进程崩溃

单个网页被 403 拒绝属于研究过程中可预期的失败。工具将错误返回模型，模型可以换来源继续；只有编程错误才应使运行失败。

### 流式输出区分模型与工具消息

`messages` 用于输出模型 token，`values` 用于观察已完成节点和保留最终 state。工具正文不直接打印，避免搜索 JSON 和整页网页淹没终端。

## 5. 代码与学习材料

- `learning/labs/01_minimal_tool_loop.py`：无网络最小 Agent Loop。
- `learning/labs/02_real_files_and_shell.py`：虚拟文件与真实 OS 能力对照。
- `learning/lesson-01.md`：第一课精读讲义。
- `learning/practice-01-real-search-agent.md`：真实搜索 Agent 逐段讲解。
- `learning/search-agent/search_agent.py`：完整实现。
- `learning/search-agent/tests/`：流式与网络失败回归测试。
- `learning/search-agent/README.md`：安装和运行入口。

## 6. 验证证据

离线回归：

```text
test_http_403_becomes_a_tool_result ... ok
test_stream_hides_tool_content_and_keeps_events ... ok
Ran 2 tests ... OK
```

真实 Aqours 查询复测：

```text
write_todos
-> web_search x2
-> fetch_url x3
-> 继续搜索与替换受限来源
-> write_file
-> write_todos
-> 最终回答
```

进程退出码为 0；网站限制被转换为工具结果，报告正常落盘。

## 7. 已知限制

- 搜索依赖第三方 HTML/RSS 结构，不如正式搜索 API 稳定。
- 不执行 JavaScript，无法完整读取动态网站。
- 当前验收统计 `fetch_url` 调用次数，尚未区分成功来源数量。
- Sources 是报告级来源列表，还没有逐条 claim-citation 映射。
- 没有 checkpoint，中断后不能恢复同一研究线程。
- 仍是单个主 Agent，尚未实现明确的 researcher、writer、reviewer 分工。
- 没有 token、延迟、搜索次数和总预算的统一控制器。

## 8. Stage 02 方向

下一阶段将围绕“自适应研究强度”展开：

1. 结构化证据账本与成功来源质量门槛。
2. LangGraph checkpoint、`thread_id` 与恢复执行。
3. `single | multi | auto` 可选 Agent 拓扑。
4. `low | medium | high | xhigh` 资源等级。
5. 依据任务复杂度、证据不足和矛盾情况自动升级，而不是只做静态模型切换。
6. 记录质量、token、延迟和工具调用，形成可比较的实验结果。

推理等级或多 Agent 开关本身不是独有创新；可验证的创新点应是：同一套质量门槛下，系统如何根据证据状态动态扩展模型、工具预算和 Agent 拓扑，并证明它比固定配置有更好的质量/成本折中。

## 9. PPT 建议页

本阶段可以直接拆成六页：

1. 问题：普通 LLM 回答缺乏真实行动和来源。
2. 三层架构：LangChain、LangGraph、Deep Agents。
3. 闭环：计划、搜索、观察、再决策、写报告。
4. 工程边界：SSRF、下载限制、错误恢复、密钥隔离。
5. 实验：Aqours 403 从整图崩溃到自动恢复。
6. 演进：结构化证据、checkpoint、多 Agent、自适应计算。
