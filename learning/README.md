# Deep Agents 中文精读与复现

这套材料的目标不是“跑通一个 API 示例”，而是逐层掌握 Deep Agents 的设计、源码和验证方法，最后能够独立修改核心能力并解释取舍。

## 当前基线

- 上游仓库：`langchain-ai/deepagents`
- 分支：`main`
- 学习起点提交：`4ddb361b99857c1fc23afb9ada0a68162c190f74`
- 主攻包：`libs/deepagents/`（Python SDK）
- 环境：Python 3.11+，依赖由 `uv` 和包内 `uv.lock` 管理

## 使用方式

每一课都按同一个闭环进行：

1. 先写下自己对运行结果的预测。
2. 只读当课列出的少量源码，不做全仓库漫游。
3. 跑一个无网络、可重复的实验。
4. 改一个变量，再跑一次，比较差异。
5. 不看讲义，用自己的话画出调用链并解释边界。

## 导航

- [完整路线](roadmap.md)
- [第 1 课：从 `create_deep_agent` 到一次工具循环](lesson-01.md)
- [学习进度](progress.md)
- [第 1 课实验](labs/01_minimal_tool_loop.py)
- [第 1 课真实文件与 shell 对照实验](labs/02_real_files_and_shell.py)
- [真实联网搜索 Agent](search-agent/README.md)
- [实践课讲义：全真实联网搜索 Agent](practice-01-real-search-agent.md)
- [阶段快照](stages/README.md)
- [Stage 01：真实单 Agent 搜索闭环](stages/stage-01-real-search-loop.md)
- [Stage 02：分级、多 Agent、可恢复研究闭环](stages/stage-02-adaptive-research.md)

## 先记住的一句话

Deep Agents 不是新的 agent runtime；它是一个“有主见的 agent harness”：在 LangChain `create_agent` 之上组装 middleware、backend、subagent、skills、memory 等能力，最终仍由 LangGraph 执行状态图。
