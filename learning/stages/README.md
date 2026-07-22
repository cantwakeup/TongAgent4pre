# 阶段快照

每个阶段快照都采用相同结构，便于复现、回溯和制作讲解 PPT：

1. 阶段目标
2. 完成能力
3. 架构与关键调用链
4. 代码与文档索引
5. 验证证据
6. 已知限制
7. 下一阶段假设

快照只记录可以从代码、测试或真实运行中验证的结果，不把计划中的能力写成已经实现。

## 快照列表

- [`stage-01-real-search-loop.md`](stage-01-real-search-loop.md)：真实单 Agent 搜索闭环
- [`stage-02-adaptive-research.md`](stage-02-adaptive-research.md)：结构化证据、持久化线程和分级多 Agent 研究
- [`stage-03a-explicit-research-state.md`](stage-03a-explicit-research-state.md)：显式研究计划、子问题状态机、结构覆盖度与节点级恢复
- [`stage-03b-search-evidence-controls.md`](stage-03b-search-evidence-controls.md)：每 SQ 保留预算、搜索引擎回退、自适应短页证据和 canonical 来源绑定
- [`stage-03c-evidence-graph.md`](stage-03c-evidence-graph.md)：exact quote 校验、Claim-Evidence-Source 图、显式冲突和严格报告映射
- [`stage-03d-adaptive-control.md`](stage-03d-adaptive-control.md)：证据缺口驱动的确定性控制、baseline/reserve 硬预算、attempt ledger、幂等恢复与 run 隔离
