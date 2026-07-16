# 真实联网搜索 Agent

这个最小项目不使用 fake model：

- LLM：私有 OpenAI-compatible API，默认 `gpt-5.4-mini`
- 搜索：真实 DuckDuckGo HTML 搜索，Bing RSS 作为后备
- 阅读：真实抓取公开 HTTP(S) 网页
- 编排：Deep Agents + LangChain + LangGraph
- 输出：真实写入 `output/report.md`

## 安装

在本目录执行：

```bash
../../../.tools/uv sync
```

环境要求：

- Python `>=3.11,<4.0`，由 uv 自动选择；
- 能访问模型 API、DuckDuckGo/Bing 和待读取的公开网页；
- 不需要 Conda、CUDA 或本地 GPU；
- 本目录的 `.venv` 与仓库核心 SDK 的 `.venv` 相互独立。

仓库开发规范要求由 uv 管理环境，不要在同一项目中再混用 Conda/pip。

复制模板并填写本地配置：

```bash
cp .env.example .env
```

`.env` 需要包含：

```text
SEARCH_AGENT_BASE_URL=<private-api-base-url>
SEARCH_AGENT_API_KEY=<private-api-key>
```

`.env` 已被 Git 忽略，程序不会打印密钥。

当前服务的免费 Qwen 后端没有启用自动 tool-choice parser，带工具请求会返回 HTTP 400，因此第一版使用支持自动工具调用的 `gpt-5.4-mini`。免费模型仍可在后续的“无工具纯总结”阶段使用。

## 运行

```bash
.venv/bin/python search_agent.py "你想研究的问题"
```

默认会实时显示模型文本、工具调用及工具结果大小。其他模式：

```bash
# 不显示实时过程
.venv/bin/python search_agent.py --no-stream "你想研究的问题"

# 流式运行，并在结束后完整打印报告
.venv/bin/python search_agent.py --print-report "你想研究的问题"
```

默认问题是：LangGraph 和 Deep Agents 的关系、各自职责，以及应该在什么场景使用它们。

## 安全边界

`fetch_url` 只允许公网 HTTP(S)，拒绝本机与内网地址，限制重定向、下载大小和返回文本长度。Agent 使用 `FilesystemBackend`，只能通过文件工具在 `output/` 的虚拟根中工作；它没有 shell 工具。

单个网页返回 403/404、超时或 DNS 失败时，`fetch_url` 会把失败原因作为工具结果交还模型，让模型改选其他来源，而不是让整个 LangGraph 工具节点退出。

这仍然不是浏览器级安全沙箱。网页内容可能包含提示注入，因此 system prompt 要求模型只把网页当资料，不把网页文字当指令。生产系统还应增加域名策略、内容隔离、审计、配额和人工审批。
