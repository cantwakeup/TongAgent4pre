"""Build a sanitized, deterministic bundle for the offline Harness Console."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SEARCH_AGENT_ROOT = REPOSITORY_ROOT / "learning" / "search-agent"
FORMAL_EXPERIMENT_ID = "final-transparent-harness-v2-formal-20260722T162330Z"
FAULT_EXPERIMENT_ID = "final-transparent-harness-v2-fault-20260722T175204Z"
DEMO_REPLAYS = (
    {
        "task_id": "0383a3ee-47a7-41a4-b493-519bdefe0488",
        "label": "顺利完成",
        "description": "1 次搜索 · 1 次抓取 · 成功回答",
    },
    {
        "task_id": "305ac316-eef6-4446-960a-92d80d542f82",
        "label": "改写查询后完成",
        "description": "低相关结果 → 调整查询 → 成功回答",
    },
    {
        "task_id": "46719c30-f4c3-4cad-be07-d5cb21eee6bb",
        "label": "预算保护停止",
        "description": "Token 预算触发 → 安全终止",
    },
)
CHECKPOINT_REPLAY = {
    "task_id": "fault-resume-alpha",
    "label": "中断后恢复",
    "description": "受控恢复实验 · 中断 → 检查点 → 继续回答",
}
DEMO_TASK_ID = str(DEMO_REPLAYS[0]["task_id"])

FORMAL_ROOT = SEARCH_AGENT_ROOT / "output" / "evaluations" / FORMAL_EXPERIMENT_ID
FAULT_ROOT = SEARCH_AGENT_ROOT / "output" / "evaluations" / FAULT_EXPERIMENT_ID


def _attempt_root(task_id: str) -> Path:
    return FORMAL_ROOT / "tongagent_standard" / task_id / "attempt-0001"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _compact_timeline(
    events: list[dict[str, Any]],
    *,
    completion_status: str,
    checkpoint_restored: bool = False,
) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    search_needs_revision = False
    previous_search_query = ""
    run_started_count = 0
    model_call_count = 0
    total_model_calls = sum(
        event.get("event_type") == "model_call_finished" for event in events
    )
    for event in events:
        event_type = str(event.get("event_type", ""))
        details = event.get("details") or event.get("payload")
        details = details if isinstance(details, dict) else {}
        budget = details.get("budget")
        budget = budget if isinstance(budget, dict) else {}

        if event_type == "run_started":
            run_started_count += 1
            if run_started_count > 1 and checkpoint_restored:
                timeline.extend(
                    [
                        {
                            "kind": "interrupt",
                            "title": "受控实验模拟进程中断",
                            "module": "evaluation/fault_injection.py",
                            "function": "_run_case()",
                            "detail": (
                                "页面抓取完成后终止第一次执行；检查点数据库和"
                                "已抓取来源保留在原运行目录。"
                            ),
                            "status": "warning",
                            "sequence": event.get("sequence"),
                        },
                        {
                            "kind": "recovery",
                            "title": "从检查点恢复执行",
                            "module": "evaluation/systems/transparent_react.py",
                            "function": "_CheckpointedGraph.invoke()",
                            "detail": (
                                "检测到检查点文件 checkpoint.sqlite，恢复同一任务状态；"
                                "已成功抓取的来源不会再次请求。"
                            ),
                            "status": "recovered",
                            "sequence": event.get("sequence"),
                        },
                    ]
                )
                continue
            timeline.append(
                {
                    "kind": "system",
                    "title": "任务已接收",
                    "module": "evaluation/systems/common.py",
                    "function": "run_graph_system()",
                    "detail": "已创建原子运行目录，轨迹与预算计数同步启动。",
                    "status": "success",
                    "sequence": event.get("sequence"),
                }
            )
        elif event_type == "runtime_prepared":
            model = details.get("model") or {}
            model_name = str(model.get("name", "model"))
            if model_name == "fixture-chat-model":
                model_name = "离线受控模型"
            tools = details.get("tools") or []
            tool_labels = {
                "web_search": "网页搜索",
                "fetch_url": "页面抓取",
            }
            timeline.append(
                {
                    "kind": "runtime",
                    "title": (
                        "恢复后的执行环境已就绪"
                        if run_started_count > 1 and checkpoint_restored
                        else "Agent 执行环境准备完成"
                    ),
                    "module": "evaluation/systems/common.py",
                    "function": "prepare_runtime()",
                    "detail": (
                        f"模型 {model_name}；工具 "
                        f"{'、'.join(tool_labels.get(str(tool), str(tool)) for tool in tools)}"
                    ),
                    "status": "success",
                    "sequence": event.get("sequence"),
                }
            )
        elif event_type == "model_call_finished":
            model_call_count += 1
            usage = details.get("token_usage") or {}
            is_final_call = (
                completion_status == "completed"
                and model_call_count == total_model_calls
            )
            stage = str(details.get("stage", ""))
            if is_final_call:
                title = f"模型调用 {model_call_count}：生成最终答案"
            elif stage == "evidence_selection":
                title = f"模型调用 {model_call_count}：检查来源并决定下一步"
            else:
                title = f"模型调用 {model_call_count}：分析任务并决定下一步"
            timeline.append(
                {
                    "kind": "model",
                    "title": title,
                    "module": "evaluation/systems/common.py",
                    "function": "EvaluationMiddleware.wrap_model_call()",
                    "detail": (
                        f"{int(usage.get('total_tokens') or 0):,} Token；"
                        f"耗时 {float(details.get('duration_seconds') or 0):.1f} 秒"
                    ),
                    "status": "success",
                    "sequence": event.get("sequence"),
                    "budget": _budget_snapshot(budget),
                }
            )
        elif event_type == "tool_call_finished":
            tool_name = str(details.get("tool_name", "tool"))
            result = details.get("result")
            result = result if isinstance(result, dict) else {}
            if tool_name == "web_search":
                outcome = str(result.get("outcome", ""))
                current_query = str(result.get("query", ""))
                low_quality = outcome in {"low_relevance", "empty_results"}
                if low_quality:
                    title = (
                        "首次搜索没有找到结果"
                        if outcome == "empty_results"
                        else "首次搜索结果不够相关"
                    )
                    display_status = "warning"
                    search_needs_revision = True
                    previous_search_query = current_query
                    detail = f"原查询：「{current_query}」"
                elif search_needs_revision:
                    title = "Agent 改写查询后重新搜索"
                    display_status = "recovered"
                    search_needs_revision = False
                    detail = (
                        f"原查询：「{previous_search_query}」 → "
                        f"新查询：「{current_query}」"
                    )
                    previous_search_query = current_query
                else:
                    title = "网页搜索"
                    display_status = str(details.get("status", "unknown"))
                    detail = f"查询：「{current_query}」"
                    previous_search_query = current_query
                provider_statuses = result.get("provider_statuses") or []
                providers = [
                    {
                        "name": str(item.get("provider", "unknown")),
                        "status": str(item.get("status", "unknown")),
                    }
                    for item in provider_statuses
                    if isinstance(item, dict)
                ]
                timeline.append(
                    {
                        "kind": "search",
                        "title": title,
                        "module": "search_agent.py → retrieval_backend.py",
                        "function": ("limited_web_search() → SearchBroker.search()"),
                        "detail": detail,
                        "status": display_status,
                        "sequence": event.get("sequence"),
                        "duration": float(details.get("duration_seconds") or 0),
                        "providers": providers,
                        "result_count": len(result.get("results") or []),
                    }
                )
            elif tool_name == "fetch_url":
                fetch_status = str(details.get("status", "unknown"))
                failure = details.get("failure")
                failure = failure if isinstance(failure, dict) else {}
                if fetch_status == "success":
                    title = "来源网页抓取成功"
                    detail = str(result.get("title", result.get("url", "")))
                    if detail.startswith("Controlled source for "):
                        detail = (
                            "受控实验来源页（"
                            f"{detail.removeprefix('Controlled source for ')}）"
                        )
                    module = "search_agent.py → retrieval_backend.py"
                    function = "limited_fetch_url() → RetrievalSession.fetch()"
                else:
                    failure_message = str(failure.get("message", "抓取失败"))
                    title = (
                        "重复抓取被安全阻止"
                        if failure_message == "same_url_already_attempted"
                        else "页面抓取失败"
                    )
                    detail = (
                        "该链接已成功抓取，安全层阻止重复请求。"
                        if failure_message == "same_url_already_attempted"
                        else failure_message
                    )
                    module = "search_agent.py"
                    function = "ResearchBudget.fetch_suppression()"
                timeline.append(
                    {
                        "kind": "fetch",
                        "title": title,
                        "module": module,
                        "function": function,
                        "detail": detail,
                        "status": (
                            "warning" if fetch_status != "success" else fetch_status
                        ),
                        "sequence": event.get("sequence"),
                        "duration": float(details.get("duration_seconds") or 0),
                        "source_id": result.get("source_id"),
                        "url": result.get("url"),
                        "content_chars": int(result.get("content_chars") or 0),
                        "acquisition_method": result.get("acquisition_method"),
                    }
                )
        elif event_type == "model_call_budget_exceeded":
            attempted = details.get("attempted")
            attempted = attempted if isinstance(attempted, dict) else {}
            timeline.append(
                {
                    "kind": "budget",
                    "title": "预算保护触发",
                    "module": "evaluation/systems/common.py",
                    "function": "EvaluationMiddleware._reserve_model_call()",
                    "detail": (
                        f"需要预留 {int(attempted.get('token_reservation') or 0):,} "
                        f"Token，当前仅剩 {int(attempted.get('available_tokens') or 0):,}"
                    ),
                    "status": "warning",
                    "sequence": event.get("sequence"),
                    "budget": _budget_snapshot(budget),
                }
            )
        elif event_type == "run_exception":
            exception_type = str(details.get("exception_type", ""))
            if exception_type == "BudgetExceeded":
                timeline.append(
                    {
                        "kind": "stop",
                        "title": "任务按预算策略安全停止",
                        "module": "evaluation/systems/common.py",
                        "function": "run_graph_system()",
                        "detail": "未生成无依据答案，已完整保存已有轨迹与运行文件。",
                        "status": "stopped",
                        "sequence": event.get("sequence"),
                    }
                )
    return timeline


def _budget_snapshot(budget: dict[str, Any]) -> dict[str, Any]:
    if not budget:
        return {}
    return {
        "tokens": int(budget.get("total_tokens") or 0),
        "model_calls": int(budget.get("model_calls") or 0),
        "search_calls": int(budget.get("search_calls") or 0),
        "fetch_calls": int(budget.get("fetch_calls") or 0),
        "elapsed_seconds": round(float(budget.get("elapsed_seconds") or 0), 1),
    }


def _artifacts(attempt_root: Path) -> list[dict[str, Any]]:
    descriptions = {
        "trace.jsonl": "模型、工具、预算与故障事件的有序轨迹",
        "checkpoint.sqlite": "LangGraph 执行检查点",
        "checkpoint_manifest.json": "恢复身份与检查点状态",
        "budget.json": "Token、工具调用与时限记账",
        "source_ledger.json": "Canonical 抓取来源记录",
        "posthoc_audit.json": "不阻塞答案的后验来源审计",
        "result.json": "最终结果与冻结评分",
    }
    paths = {
        "trace.jsonl": attempt_root / "native" / "trace.jsonl",
        "checkpoint.sqlite": attempt_root / "native" / "checkpoint.sqlite",
        "checkpoint_manifest.json": (
            attempt_root / "native" / "checkpoint_manifest.json"
        ),
        "budget.json": attempt_root / "native" / "budget.json",
        "source_ledger.json": attempt_root / "native" / "source_ledger.json",
        "posthoc_audit.json": attempt_root / "native" / "posthoc_audit.json",
        "result.json": attempt_root / "result.json",
    }
    return [
        {
            "name": name,
            "description": descriptions[name],
            "bytes": path.stat().st_size,
            "status": "persisted" if path.is_file() else "missing",
        }
        for name, path in paths.items()
    ]


def _formal_benchmark(summary: dict[str, Any]) -> list[dict[str, Any]]:
    labels = {
        "bare_simple_react": "Bare Simple ReAct",
        "tongagent_standard": "TongAgent Standard",
        "vanilla_deepagents": "Vanilla DeepAgents",
    }
    rows = []
    for system in summary["systems"]:
        system_id = str(system["system_id"])
        failures = system.get("failure_distribution") or {}
        rows.append(
            {
                "system_id": system_id,
                "label": labels[system_id],
                "runs": int(system["runs"]),
                "answer_rate": float(system["answer_rate"]),
                "raw_em": float(system["raw_whole_string_em_rate"]),
                "standard_em": float(system["standard_normalized_em_rate"]),
                "runner_errors": int(failures.get("runner_error") or 0),
                "budget_exhausted": int(failures.get("budget_exhausted") or 0),
                "mean_wall": round(float(system["mean_wall_time_seconds"]), 1),
            }
        )
    return rows


def _fault_scenarios(fault_summary: dict[str, Any]) -> list[dict[str, Any]]:
    labels = {
        "fetch_first_attempt_failure": "首次抓取失败",
        "temporary_model_connection_failure": "模型连接暂时中断",
        "process_interruption_and_resume": "进程中断与恢复",
        "near_budget_exhaustion": "接近预算上限",
    }
    grouped: dict[str, dict[str, Any]] = {}
    for row in fault_summary["rows"]:
        fault = str(row["fault"])
        entry = grouped.setdefault(
            fault,
            {"id": fault, "label": labels[fault], "systems": {}},
        )
        entry["systems"][str(row["system_id"])] = {
            "recovery_success": bool(row["recovery_success"]),
            "valid_terminal": bool(row["valid_terminal_result"]),
            "checkpoint_restored": bool(row["checkpoint_restored"]),
            "duplicate_fetch_calls": int(row["duplicate_fetch_calls"]),
            "trace_complete": bool(row["trace_complete"]),
            "total_tokens": row["total_tokens"],
            "wall_time_seconds": round(float(row["wall_time_seconds"]), 2),
        }
    return [grouped[fault] for fault in labels]


def _build_replay(
    spec: dict[str, str],
    *,
    attempt_root: Path | None = None,
    checkpoint_replay: bool = False,
) -> dict[str, Any]:
    task_id = str(spec["task_id"])
    attempt_root = attempt_root or _attempt_root(task_id)
    task_path = attempt_root / "task.json"
    if not task_path.is_file():
        task_path = attempt_root / "native" / "task.json"
    task = _load_json(task_path)
    result = _load_json(attempt_root / "result.json")
    audit = _load_json(attempt_root / "native" / "posthoc_audit.json")
    checkpoint = _load_json(attempt_root / "native" / "checkpoint_manifest.json")
    source_ledger = _load_json(attempt_root / "native" / "source_ledger.json")
    trace_events = _load_jsonl(attempt_root / "native" / "trace.jsonl")
    resolved_budget = result["resolved_config"]["budget"]
    checkpoint_restored = bool(checkpoint["checkpoint_restored"])

    return {
        "id": task_id,
        "label": spec["label"],
        "description": spec["description"],
        "replay_kind": (
            "controlled_checkpoint_recovery"
            if checkpoint_replay
            else "formal_benchmark"
        ),
        "artifact_directory": attempt_root.relative_to(REPOSITORY_ROOT).as_posix(),
        "run": {
            "question": task["question"],
            "model_label": (
                "离线受控模型"
                if result["resolved_config"]["model"]["name"] == "fixture-chat-model"
                else result["resolved_config"]["model"]["name"]
            ),
            "system_id": result["system_id"],
            "completion_status": result["completion_status"],
            "answer_status": result["answer_status"],
            "raw_model_answer": result["raw_model_answer"],
            "final_answer": result["final_answer"],
            "answer_unchanged": bool(audit["answer_unchanged"]),
            "raw_em": bool(result["raw_whole_string_em"]),
            "standard_em": bool(result["standard_normalized_em"]),
            "metrics": {
                "tokens": int(result["token_usage"]["total_tokens"]),
                "input_tokens": int(result["token_usage"]["input_tokens"]),
                "output_tokens": int(result["token_usage"]["output_tokens"]),
                "cached_input_tokens": int(
                    result["token_usage"].get("cached_input_tokens") or 0
                ),
                "model_calls": sum(
                    event.get("event_type") == "model_call_finished"
                    for event in trace_events
                ),
                "search_calls": int(result["search_calls"]),
                "fetch_calls": int(result["fetch_calls"]),
                "wall_time_seconds": round(float(result["wall_time_seconds"]), 1),
            },
            "limits": {
                "tokens": int(resolved_budget["max_total_tokens"]),
                "model_calls": int(resolved_budget["max_model_calls"]),
                "search_calls": int(resolved_budget["max_search_calls"]),
                "fetch_calls": int(resolved_budget["max_fetch_calls"]),
                "wall_time_seconds": float(resolved_budget["wall_time_seconds"]),
            },
            "checkpoint": {
                "file": checkpoint["checkpoint_file"],
                "restored": checkpoint_restored,
                "thread_id": checkpoint["thread_id"],
            },
        },
        "timeline": _compact_timeline(
            trace_events,
            completion_status=str(result["completion_status"]),
            checkpoint_restored=checkpoint_replay and checkpoint_restored,
        ),
        "sources": source_ledger["successful_fetches"],
        "artifacts": _artifacts(attempt_root),
    }


def build_bundle() -> dict[str, Any]:
    """Build the complete browser-facing payload from frozen artifacts."""

    formal_summary = _load_json(FORMAL_ROOT / "summary.json")
    fault_summary = _load_json(FAULT_ROOT / "summary.json")
    replays = [_build_replay(spec) for spec in DEMO_REPLAYS]
    replays.append(
        _build_replay(
            CHECKPOINT_REPLAY,
            attempt_root=(
                FAULT_ROOT
                / "tongagent_standard"
                / str(CHECKPOINT_REPLAY["task_id"])
                / "attempt-0001"
            ),
            checkpoint_replay=True,
        )
    )
    default_replay = replays[0]
    return {
        "schema_version": 1,
        "provenance": {
            "label": "正式实验与受控恢复轨迹",
            "experiment_id": FORMAL_EXPERIMENT_ID,
            "git_sha": formal_summary["git_sha"],
            "task_id": DEMO_TASK_ID,
            "replay_task_ids": [replay["id"] for replay in replays],
            "fault_experiment_id": FAULT_EXPERIMENT_ID,
            "contains_reference_answer": False,
            "network_required": False,
        },
        "parity": {
            "model": "GPT-5.5",
            "seed": 17,
            "prompt_hash": (
                "73a4e7cc90c0ed0b2f5f743de28f35747647fc33c80e1bb6af706b1e5ada6eef"
            ),
            "fairness_fingerprint": formal_summary["fairness_fingerprint"],
            "tools": ["web_search", "fetch_url"],
            "policy": "Shared transparent ReAct",
        },
        "run": default_replay["run"],
        "timeline": default_replay["timeline"],
        "sources": default_replay["sources"],
        "artifacts": default_replay["artifacts"],
        "replays": replays,
        "benchmark": {
            "label": "冻结的 48 次正式运行",
            "systems": _formal_benchmark(formal_summary),
            "claims": {
                "accuracy_superiority": False,
                "performance_non_inferiority": False,
                "operational_reliability_superiority": True,
            },
        },
        "fault_lab": {
            "label": "受控离线故障注入",
            "api_cost_usd": 0,
            "systems": fault_summary["systems"],
            "scenarios": _fault_scenarios(fault_summary),
        },
        "architecture": [
            {
                "index": "01",
                "name": "共享 ReAct 策略",
                "detail": "提示词、模型参数、工具和原始答案保持一致。",
            },
            {
                "index": "02",
                "name": "统一检索后端",
                "detail": "Provider 切换、Canonical URL、缓存和 Host 冷却。",
            },
            {
                "index": "03",
                "name": "执行预算控制",
                "detail": "统一管理 Token、模型调用、工具调用和业务时限。",
            },
            {
                "index": "04",
                "name": "透明执行框架",
                "detail": "检查点、失败重试、原子 Artifact 与持久化轨迹。",
            },
            {
                "index": "05",
                "name": "后验审计",
                "detail": "来源账本和证据映射只做审计，绝不改写答案。",
            },
        ],
    }


def _validate(bundle: dict[str, Any]) -> None:
    if bundle["provenance"]["contains_reference_answer"]:
        raise ValueError("demo bundle must not contain a reference answer")
    if len(bundle["replays"]) != 4:
        raise ValueError(
            "demo must retain three formal replays and one recovery replay"
        )
    for replay in bundle["replays"]:
        if not replay["artifact_directory"]:
            raise ValueError(f"replay {replay['id']} is missing artifact provenance")
        if replay["run"]["raw_model_answer"] != replay["run"]["final_answer"]:
            raise ValueError(
                f"replay {replay['id']} violates transparent-answer parity"
            )
        if len(replay["timeline"]) < 7:
            raise ValueError(f"replay {replay['id']} timeline is incomplete")
        for event in replay["timeline"]:
            if not event.get("module") or not event.get("function"):
                raise ValueError(
                    f"replay {replay['id']} event {event.get('sequence')} "
                    "is missing code origin"
                )
            if (
                event.get("kind") == "fetch"
                and event.get("status") == "success"
                and not str(event.get("url", "")).startswith(("http://", "https://"))
            ):
                raise ValueError(
                    f"replay {replay['id']} has a fetch event without a source URL"
                )
        if any(item["status"] != "persisted" for item in replay["artifacts"]):
            raise ValueError(
                f"replay {replay['id']} is missing one or more canonical artifacts"
            )
    if not bundle["replays"][3]["run"]["checkpoint"]["restored"]:
        raise ValueError("checkpoint recovery replay did not restore its checkpoint")
    if len(bundle["fault_lab"]["scenarios"]) != 4:
        raise ValueError("fault demo must retain all four preregistered scenarios")
    if len(bundle["benchmark"]["systems"]) != 3:
        raise ValueError("formal benchmark must retain all three systems")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "demo_bundle.json",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    bundle = build_bundle()
    _validate(bundle)
    rendered = json.dumps(bundle, ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if not args.output.is_file():
            raise FileNotFoundError(args.output)
        if args.output.read_text(encoding="utf-8") != rendered:
            raise ValueError(f"demo bundle is stale: {args.output}")
        print(f"demo bundle valid: {args.output}")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
