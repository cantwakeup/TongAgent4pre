# TongAgent

TongAgent 是一个建立在 LangChain、LangGraph 和 Deep Agents 之上的可演进研究 Agent 项目，同时也是一套中文源码精读、复现与工程化记录。

当前阶段已经完成一个真实联网搜索闭环：模型规划研究任务、调用搜索和网页读取工具、根据工具结果继续决策、生成报告，并保存可审计的工具轨迹。

## 当前里程碑

- `stage-01`：单 Agent 真实联网研究闭环
- 下一阶段：结构化证据、checkpoint、可选多 Agent、资源与推理强度分级

项目入口与复现方式见 [`learning/README.md`](learning/README.md)，阶段快照见 [`learning/stages/`](learning/stages/)。

## 上游来源

本项目从 `langchain-ai/deepagents` 的提交 `4ddb361b99857c1fc23afb9ada0a68162c190f74` 开始学习和开发。上游代码遵循仓库中的 MIT License；TongAgent 自己的增量集中记录在 `learning/`。
