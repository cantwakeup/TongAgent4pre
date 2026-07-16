# 学习进度

## 2026-07-15

- [x] 获取上游 `main` 的核心 SDK、测试和代表性示例
- [x] 阅读根 README、`AGENTS.md`、`libs/ARCHITECTURE.md`、`libs/DEVELOPMENT.md`
- [x] 定位公共入口 `create_deep_agent`
- [x] 建立十课路线与第 1 课讲义
- [x] 编写第 1 课无 API Key 实验
- [x] 完成核心运行依赖同步（52 个锁定包）
- [x] 跑通第 1 课无 API Key 实验
- [x] 跑通第 1 课真实文件与 shell 对照实验
- [x] 实现并跑通全真实联网搜索 Agent
- [x] 验证搜索、网页读取、模型 tool calling、报告落盘与 SSRF 拒绝
- [x] 增加模型 token、工具事件的流式输出
- [x] 修复单个网页 403 导致整个工具节点退出的问题
- [x] 增加流式内容隔离和 HTTP 403 回归测试
- [x] 建立 Stage 01 可复现快照与 PPT 讲解索引

## 2026-07-16

- [x] 增加 `[S#]` 结构化来源账本、成功来源门槛和失败来源记录
- [x] 增加 `single | multi | auto` 可选拓扑与 researcher/reviewer 明确角色
- [x] 增加 `low | medium | high | xhigh` 四档硬资源策略
- [x] 完成 nano single、nano multi、mini + 免费 reviewer 的真实路径验证
- [x] 增加 SQLite checkpoint、命名 thread 恢复与本次 trace 历史隔离
- [x] 在 `report.md` 之外保存 `trace.json`、`sources.json`、`run.json` 审计产物
- [x] 建立 Stage 02 可复现快照与创新边界说明
- [x] 运行搜索 Agent 的 10 个离线回归测试
- [ ] 安装上游完整 test group 并运行整个 monorepo 测试（后续按需进行）
- [ ] 用户完成第 1 课口头自测和四个修改练习

## 环境记录

- 系统 Python：3.13.0，满足项目 `>=3.11,<4.0`
- 系统 `/snap/bin/uv` 在当前容器缺少所需权限，无法运行
- 已审查官方 uv 0.11.28 安装脚本，并限定安装到工作区 `.tools/`
- 官方 uv 0.11.28 已下载、通过官方 SHA-256 校验，并安装到工作区 `.tools/`
- 基础环境位于 `libs/deepagents/.venv`；deepagents 0.6.12、LangChain 1.3.12 已验证可导入
- 外网到 `files.pythonhosted.org` 较慢；`--frozen` 使用锁文件中的固定 wheel URL，设置镜像不会重写这些 URL
- 为避免第一课过度准备，暂未同步包含 av、ruff、ty、pytest 等包的完整 test group
