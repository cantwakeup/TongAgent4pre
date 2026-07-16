# TongAgent

TongAgent 是一个建立在 LangChain、LangGraph 和 Deep Agents 之上的可演进研究 Agent 项目，同时也是一套中文源码精读、复现与工程化记录。

当前阶段已经完成一个有显式研究计划、可分级、可恢复、可选择 single/multi 拓扑的真实联网研究闭环：外层 LangGraph 按子问题执行研究与结构覆盖判断，内层 Deep Agent 收集证据并生成报告，同时保存可审计的计划、状态事件、工具轨迹、来源账本和 SQLite checkpoint。

## 当前里程碑

- `stage-01`：单 Agent 真实联网研究闭环
- `stage-02`：结构化证据、SQLite checkpoint、可选多 Agent、资源与研究强度分级
- `stage-03a`：显式研究计划、子问题状态机、结构覆盖度与节点级恢复
- 下一阶段：正文证据片段、可信度、Claim-Evidence-URL 映射与冲突对象

项目入口与复现方式见 [`learning/README.md`](learning/README.md)，阶段快照见 [`learning/stages/`](learning/stages/)。

## 上游来源

本项目从 `langchain-ai/deepagents` 的提交 `4ddb361b99857c1fc23afb9ada0a68162c190f74` 开始学习和开发。上游代码遵循仓库中的 MIT License；TongAgent 自己的增量集中记录在 `learning/`。
