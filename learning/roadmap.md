# 学习路线

## 第一阶段：建立主干心智模型

### 第 1 课：一次完整工具循环

- 入口：`deepagents.create_deep_agent`
- 搞清三层关系：Deep Agents / LangChain / LangGraph
- 用假模型复现：模型请求工具 → 工具执行 → 状态更新 → 模型结束
- 验收：能脱离讲义画出一次 invoke 的主调用链

### 第 2 课：精读组装器 `graph.py`

- 模型与 profile 的解析
- 主 agent 与 general-purpose subagent 的构造
- 默认 middleware 的顺序及顺序为何重要
- prompt 的 prefix / base / suffix 合成
- 验收：给定一组参数，能预测最终工具面和 middleware 栈

### 第 3 课：Middleware 机制

- `before_model`、`wrap_model_call`、`wrap_tool_call` 的职责差异
- Todo、Filesystem、PatchToolCalls 的最小实现
- 自定义 middleware 如何替换同名默认项
- 验收：写一个小 middleware，并用单元测试证明它在正确阶段生效

## 第二阶段：掌握 Deep Agents 的核心能力

### 第 4 课：Backend 与文件系统

- `BackendProtocol` 是能力边界
- `StateBackend`、`FilesystemBackend`、`StoreBackend`、`CompositeBackend`
- 虚拟路径、持久化范围与 `execute` 能力
- 验收：同一文件工具实验切换两个 backend，解释状态和持久性的变化

### 第 5 课：Subagents 与上下文隔离

- declarative、compiled、async subagent 的差别
- `task` 工具如何路由
- 私有 state 为什么不能泄漏给 subagent
- 验收：实现 researcher + writer 两个角色并验证上下文隔离

### 第 6 课：长上下文管理

- summarization、message eviction、tool output offload
- `DeepAgentState` 与 `DeltaChannel` 如何把 checkpoint 增长从 O(N²) 降到 O(N)
- 验收：构造长消息序列，观察压缩前后的状态变化

### 第 7 课：Skills、Memory 与 Human-in-the-loop

- skills 是按需加载的行为说明，不是普通工具
- thread state 与跨 thread store 的区别
- filesystem permission 与 interrupt 的合并规则
- 验收：创建一个 skill、一条持久记忆和一个需审批的写操作

## 第三阶段：从会用到会改

### 第 8 课：代表性应用复现

- 精读 `examples/deep_research`
- 先用假模型复现结构，再接入用户选择的真实模型
- 加 tracing，区分模型问题、工具问题和 orchestration 问题
- 验收：得到可重复运行的研究 agent，并保留运行记录

### 第 9 课：测试、评测与安全

- unit / integration / smoke / benchmark 的分工
- fake model 为什么比真实 API 更适合框架单测
- “trust the LLM”安全模型：边界必须落在工具与 sandbox
- 验收：为一个行为变更补回归测试，并做威胁分析

### 第 10 课：Monorepo 其他主要包

- `libs/code`：预制终端 coding agent
- `libs/cli`：init / dev / deploy
- `libs/acp`：Agent Client Protocol
- `libs/evals`：评测与 Harbor 集成
- `libs/partners`：sandbox/provider 集成
- 验收：能说明各包与核心 SDK 的依赖方向，不混淆产品层与运行时层

### 毕业项目

完成一个带以下能力的垂直 agent：自定义工具、文件 backend、一个 subagent、一个 skill、持久 memory、审批边界、无网络单测和一组真实任务评测。最后提交一份架构说明，明确哪些能力属于 Deep Agents、LangChain 和 LangGraph。
