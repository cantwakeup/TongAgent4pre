# TongAgent4pre

TongAgent4pre 是一个建立在 LangChain、LangGraph 与 Deep Agents 之上的研究型
Agent 原型。项目目标不是单纯增加 Agent 数量，而是把长程网页研究变成可恢复、
可审计的闭环：显式拆分子问题，登记 Claim–Evidence–Source 证据链，用确定性
控制器按证据缺口释放预算，并保存计划、轨迹、控制决策和验收结果。

当前实现已到
[Stage 03D：证据驱动的自适应控制](learning/stages/stage-03d-adaptive-control.md)；
`codex/benchmark-readiness` 分支正在把原型收敛为可安装、可离线测试，并能在
公平预算下统一运行 B1（Simple ReAct）、B2（原生 Deep Agents）和
B3（TongAgent）的评测版本。离线 fixture 只用于 smoke 和框架回归，不能称为
正式 benchmark 结果。

```text
问题
  -> 显式 ResearchPlan / 原子子问题
  -> single 或 multi research
  -> 搜索、抓取、Claim–Evidence–Source 账本
  -> fixed 或 deterministic adaptive controller
  -> 受约束的报告合成与结构验收
  -> checkpoint + JSON/JSONL/Markdown 运行产物
```

## 安装与离线 smoke

要求 Python `>=3.11,<4.0` 和
[`uv`](https://docs.astral.sh/uv/getting-started/installation/)。默认从标准
PyPI 解析依赖；仓库内的 Deep Agents 源码以 editable 方式安装。

```bash
cd learning/search-agent
uv sync --group test
```

先用两个无网络、无付费模型调用的关键测试做低成本 smoke：

```bash
uv run --group test python -m pytest -q \
  --disable-socket --allow-unix-socket \
  tests/test_metric_semantics.py tests/test_adaptive_control.py
```

完整离线验证从仓库根目录统一运行：

```bash
bash scripts/check_tongagent.sh
```

真实联网研究 Agent 的配置、运行和产物说明见
[learning/search-agent/README.md](learning/search-agent/README.md)。正式 baseline
不会由单次 `search_agent.py` 演示得出；Benchmark Ready Sprint 完成后，以统一
B1/B2/B3 runner 和 `BENCHMARKING.md` 为正式评测入口。

## 上游基线

TongAgent4pre 不是从零实现 Agent runtime。仓库导入并固定了
[`langchain-ai/deepagents`](https://github.com/langchain-ai/deepagents) 的
`deepagents==0.6.12` 源码快照；对应上游提交为
[`4ddb361b99857c1fc23afb9ada0a68162c190f74`](https://github.com/langchain-ai/deepagents/commit/4ddb361b99857c1fc23afb9ada0a68162c190f74)。
下面保留上游 Deep Agents 的项目介绍和使用资料。

---

<div align="center">
  <a href="https://docs.langchain.com/oss/python/deepagents/overview#deep-agents-overview">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset=".github/images/logo-dark.svg">
      <source media="(prefers-color-scheme: light)" srcset=".github/images/logo-light.svg">
      <img alt="Deep Agents Logo" src=".github/images/logo-dark.svg" width="50%">
    </picture>
  </a>
</div>

<div align="center">
  <h3>The batteries-included agent harness.</h3>
</div>

<div align="center">
  <a href="https://opensource.org/licenses/MIT" target="_blank"><img src="https://img.shields.io/pypi/l/deepagents" alt="PyPI - License"></a>
  <a href="https://pypistats.org/packages/deepagents" target="_blank"><img src="https://img.shields.io/pepy/dt/deepagents" alt="PyPI - Downloads"></a>
  <a href="https://pypi.org/project/deepagents/#history" target="_blank"><img src="https://img.shields.io/pypi/v/deepagents?label=%20" alt="Version"></a>
  <a href="https://x.com/langchain_oss" target="_blank"><img src="https://img.shields.io/twitter/url/https/twitter.com/langchain_oss.svg?style=social&label=Follow%20%40LangChain" alt="Twitter / X"></a>
</div>

<br>

Deep Agents is an open source agent harness — an opinionated agent that runs out of the box. Extend, override, or replace any piece.

**Principles:**

- **Opinionated** — defaults tuned for long-horizon, multi-step work
- **Extensible** — override or replace any piece without forking
- **Model-agnostic** — works with any LLM that supports tool calling: frontier, open-weight, or local
- **Production-ready** — built on LangGraph (streaming, persistence, checkpointing) with first-class tracing, evaluation, and deployment via LangSmith

**Features include:**

- **Sub-agents** — delegate tasks to agents with isolated context windows
- **Filesystem** — read, write, edit, or search over pluggable local, sandboxed, or remote backends
- **Context management** — summarize long threads and offload tool outputs to disk
- **Shell access** — run commands in your sandbox of choice
- **Persistent memory** — pluggable state and store backends for cross-session recall
- **Human-in-the-loop** — approve, edit, or reject tool calls before they run
- **Skills** — reusable behaviors the agent can load on demand
- **Tools** — bring your own functions or any MCP server

Deep Agents is available as a JavaScript/TypeScript library — see [deepagents.js](https://github.com/langchain-ai/deepagentsjs).

> [!NOTE]
> **Deep Agents Code** — a pre-built coding agent in your terminal, similar to Claude Code or Cursor, powered by any LLM. Install with `curl -LsSf https://langch.in/dcode | bash`. See the [documentation](https://docs.langchain.com/deepagents-code) for the full feature set.

## Quickstart

```bash
uv add deepagents
```

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model="openai:gpt-5.5",
    tools=[my_custom_tool],
    system_prompt="You are a research assistant.",
)
result = agent.invoke({"messages": "Research LangGraph and write a summary"})
```

The agent can plan, read/write files, and manage its own context. Add your own tools, swap models, customize prompts, configure sub-agents, and more. See the [documentation](https://docs.langchain.com/oss/python/deepagents/overview) for full details.

> [!TIP]
> For developing, debugging, and deploying AI agents and LLM applications, see [LangSmith](https://docs.langchain.com/langsmith/home).

## FAQ

### How is this different from LangGraph or LangChain?

LangGraph is the graph runtime. LangChain's `create_agent` is a minimal agent harness on top of it. Deep Agents is a more opinionated harness on top of `create_agent` — same building blocks, but with filesystem, sub-agents, context management, and skills bundled in. For how the three relate, see the [LangChain ecosystem overview](https://docs.langchain.com/oss/python/concepts/products).

### Does this work with open-weight or local models?

Yes. Any model that supports tool calling works — frontier APIs (OpenAI, Anthropic, Google), open-weight models hosted on providers like Baseten or Fireworks, and self-hosted models via Ollama, vLLM, or llama.cpp. Use any [LangChain chat model](https://docs.langchain.com/oss/python/langchain/models).

### Can I use this in production?

Yes! Deep Agents is built on LangGraph, designed for production agent deployments. Pair it with [LangSmith](https://docs.langchain.com/langsmith/home) for tracing, evaluation, and monitoring. See [Going to production](https://docs.langchain.com/oss/python/deepagents/going-to-production) for the full guide.

### When should I use Deep Agents vs. LangChain or LangGraph directly?

All three are layers in the same stack — see the [LangChain ecosystem overview](https://docs.langchain.com/oss/python/concepts/products) for how they relate. Use **Deep Agents** when you want the full harness — planning, context management, delegation — out of the box. Use [**LangChain's `create_agent`**](https://docs.langchain.com/oss/python/langchain/agents) when you want a lighter harness without the bundled middleware. Drop to [**LangGraph**](https://docs.langchain.com/oss/python/langgraph/overview) when the agent loop itself isn't the right shape and you need a custom graph.

The layers compose: any LangGraph `CompiledStateGraph` can be passed in as a sub-agent to a Deep Agent, so custom orchestration plugs in alongside the harness's defaults.

---

## Resources

- [Examples](examples/) — working agents and patterns
- [Documentation](https://docs.langchain.com/oss/python/deepagents/overview) — conceptual overviews and guides
- [LangChain ecosystem overview](https://docs.langchain.com/oss/python/concepts/products) — how Deep Agents, LangChain, LangGraph, and LangSmith fit together
- [API reference](https://reference.langchain.com/python/deepagents/) — complete reference for all public classes, functions, and types
- [Discussions](https://forum.langchain.com/c/oss-product-help-lc-and-lg/deep-agents/18) — community forum for technical questions, ideas, and feedback
- [LangChain Academy](https://academy.langchain.com/) — Comprehensive, free courses on LangChain libraries and products, made by the LangChain team.
- [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview) — how to contribute and find good first issues
- [Code of Conduct](https://github.com/langchain-ai/langchain/?tab=coc-ov-file) — community guidelines and standards

---

## Acknowledgements

Inspired by Claude Code: an attempt to identify what makes it general-purpose, and push that further.

## Security

Deep Agents follows a "trust the LLM" model. The agent can do anything its tools allow. Enforce boundaries at the tool/sandbox level, not by expecting the model to self-police. See the [security policy](https://github.com/langchain-ai/deepagents?tab=security-ov-file) for more information.
