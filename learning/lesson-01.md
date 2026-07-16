# 第 1 课：从 `create_deep_agent` 到一次工具循环

## 本课目标

学完后，你应该能够回答：

1. Deep Agents、LangChain、LangGraph 各自负责什么？
2. `create_deep_agent()` 返回的究竟是什么？
3. 模型发出 `write_file` 后，谁执行工具、谁保存结果、谁决定再次调用模型？
4. 为什么框架测试应优先使用 fake model？

## 三层结构

```text
Deep Agents
  负责：默认 prompt、middleware 栈、backend、subagent、skills、memory
              │ 组装并调用
              ▼
LangChain create_agent
  负责：model + tools + middleware -> 标准 agent loop
              │ 编译为
              ▼
LangGraph
  负责：状态图执行、消息状态、checkpoint、stream、interrupt、resume
```

关键判断：如果你要改“默认有哪些能力”，先看 Deep Agents；如果你要改“模型与工具怎样循环”，看 LangChain；如果你要改“状态如何持久化、暂停和恢复”，看 LangGraph。

## 公共入口

`libs/deepagents/deepagents/__init__.py` 把 `create_deep_agent` 导出为公共 API。真正的组装发生在 `libs/deepagents/deepagents/graph.py`。

主 agent 的默认组装顺序可以先简化为：

```text
TodoList
-> Skills（可选）
-> Filesystem
-> SubAgent（存在 subagent 时）
-> Summarization
-> PatchToolCalls
-> AsyncSubAgent（可选）
-> Profile 扩展
-> Prompt caching
-> Memory（可选）
-> Human approval（可选）
-> Tool exclusion（可选，最后执行）
-> LangChain create_agent(...)
```

顺序不是装饰。例如 tool exclusion 放在最后，是为了防止自定义 middleware 又把被排除的工具加回来；memory 放在 prompt cache 后部，是为了减少动态记忆导致的缓存失效。

## 本课实验的完整因果链

实验中的 fake model 不“思考”。它按预设顺序返回两条 `AIMessage`：

1. 第一条要求调用 `write_file`。
2. 第二条返回最终文本。

运行时链路如下：

```text
HumanMessage
-> LangGraph 进入 model 节点
-> fake model 返回 write_file tool call
-> LangGraph 进入 tools 节点
-> FilesystemMiddleware 提供的 write_file 被执行
-> StateBackend 产生 files 状态更新
-> ToolMessage 追加到 messages
-> LangGraph 再次进入 model 节点
-> fake model 返回最终文本
-> 图结束，返回包含 messages 与 files 的状态
```

这验证的是 orchestration，而不是模型质量。真实 API 会引入网络、鉴权、采样和模型版本变化，不适合做第一条框架因果链。

## 动手步骤

在 `libs/deepagents/` 中运行：

```bash
../../../.tools/uv sync --frozen --no-group test
PYTHONPATH=. .venv/bin/python ../../learning/labs/01_minimal_tool_loop.py
```

第一课不需要完整 test group。到测试课再同步它，并运行对应的仓库测试：

```bash
../../../.tools/uv run --group test pytest \
  tests/unit_tests/test_file_system_tools.py \
  -k 'parallel_write_file_calls or edit_file_single_replacement' \
  --disable-socket --allow-unix-socket --no-cov
```

## 必做练习

不要先看答案，依次修改实验：

1. 把 `write_file` 的路径改成 `/notes/day1.md`，预测 `result["files"]` 的 key。
2. 在第一次工具调用后增加一次 `edit_file`，把 `hello` 改成 `hello deep agents`。
3. 故意编辑不存在的文件，找到对应 `ToolMessage`，解释为什么图仍能继续。
4. 删除最终 `AIMessage`，观察异常；解释 fake model 的响应数量为什么必须覆盖所有 model 节点访问。

## 自测题

1. `StateBackend` 保存的是宿主机文件，还是 graph state 中的虚拟文件？
2. `write_file` 是调用者通过 `tools=` 传入的吗？
3. `create_deep_agent()` 为什么返回 `CompiledStateGraph`，而不是一个自定义 `DeepAgent` 类？
4. 没有 sandbox 能力的 backend 为什么不应该真的执行 shell？
5. checkpoint 与 backend 文件持久化是不是同一件事？

合格标准：不看源码，能用两分钟讲清上面的因果链；再打开 `graph.py`，能指出组装阶段与执行阶段的边界。

## 首次基线结果

2026-07-15 已在 Python 3.13 上跑通：

```text
tools:
delete, edit_file, glob, grep, ls, read_file, task, write_file, write_todos

messages:
HumanMessage -> AIMessage -> ToolMessage -> AIMessage

files:
/hello.txt -> hello from lesson 1
```

注意工具列表没有 `execute`。实验使用的 `StateBackend` 只把虚拟文件保存在 graph state，不实现 sandbox shell 协议；因此 Deep Agents 不向模型暴露真实命令执行能力。这是 backend 作为安全与能力边界的一个直接证据。

## 对照实验：真实文件与真实 shell

虚拟实验通过后，可以把 backend 换成 `LocalShellBackend`：

```bash
PYTHONPATH=. .venv/bin/python ../../learning/labs/02_real_files_and_shell.py
```

它会依次完成：

1. 用 `write_file` 把 `/real.txt` 映射并写入专用练习目录。
2. 用 `execute` 在真实宿主机运行固定的 `pwd`、`ls`、`sed` 命令。
3. 从磁盘重新读取文件并断言内容，而不是只相信 agent 返回的 state。

真实文件保存在 `learning/.runtime/lesson-01-real/real.txt`，该目录已被 Git 忽略。

`LocalShellBackend` 并不提供真正的 sandbox。`virtual_mode=True` 只能约束继承来的文件工具，无法阻止 shell 命令访问练习目录之外的路径。本实验之所以可控，是因为 fake model 的命令完全固定、环境变量不继承、超时只有 5 秒；接入真实模型前应改用容器、VM 或远程 sandbox，并为危险工具配置人工审批。
