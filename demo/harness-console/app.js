"use strict";

const state = {
  bundle: null,
  timelineIndex: 0,
  replayTimer: null,
  selectedReplay: 0,
};

const icons = {
  system: "启",
  runtime: "构",
  model: "模",
  search: "搜",
  fetch: "取",
  budget: "限",
  stop: "停",
  interrupt: "断",
  recovery: "复",
};

const formatNumber = (value) => new Intl.NumberFormat("en-US").format(value || 0);
const ratioPercent = (value, limit) => Math.min(100, Math.round((value / limit) * 100));

function el(id) {
  return document.getElementById(id);
}

function activeReplay() {
  if (state.bundle.replays && state.bundle.replays.length) {
    return state.bundle.replays[state.selectedReplay];
  }
  return {
    run: state.bundle.run,
    timeline: state.bundle.timeline,
    sources: state.bundle.sources,
    artifacts: state.bundle.artifacts,
  };
}

function escapeHtml(value) {
  return String(value == null ? "" : value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

async function loadBundle() {
  const response = await fetch("data/demo_bundle.json", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Unable to load demo bundle (${response.status})`);
  }
  return response.json();
}

function renderHeader(bundle) {
  el("provenance-label").textContent = bundle.provenance.label;
  el("experiment-short").textContent = bundle.provenance.experiment_id;
  el("question").textContent = activeReplay().run.question;
  el("question").title = activeReplay().run.question;
  el("git-sha").textContent = `git · ${bundle.provenance.git_sha.slice(0, 12)}`;
  renderParity(activeReplay());
}

function renderParity(replay) {
  const controlled = replay.replay_kind === "controlled_checkpoint_recovery";
  const chips = controlled
    ? [
        "模式 · 受控故障实验",
        `模型 · ${replay.run.model_label}`,
        "检查点 · 已恢复",
        "重复抓取 · 0",
      ]
    : [
        `模型 · ${state.bundle.parity.model}`,
        `种子 · ${state.bundle.parity.seed}`,
        "提示词一致",
        "工具一致",
        "预算一致",
      ];
  el("parity-strip").innerHTML = chips
    .map((chip) => `<span class="parity-chip">${escapeHtml(chip)}</span>`)
    .join("");
}

function renderReplayRecord(replay) {
  el("trace-source-path").textContent = replay.artifact_directory;
  el("record-sources").textContent = `${replay.sources.length} 个来源`;
  if (replay.run.completion_status === "budget_exhausted") {
    el("record-score").textContent = "预算终态";
  } else {
    const raw = replay.run.raw_em ? "✓" : "✕";
    const standard = replay.run.standard_em ? "✓" : "✕";
    el("record-score").textContent = `原始 ${raw} · 标准 ${standard}`;
  }
}

function renderReplaySelector() {
  const replays = state.bundle.replays || [];
  el("trace-selector").innerHTML = replays
    .map((replay, index) => `
      <button class="trace-choice ${index === state.selectedReplay ? "active" : ""}"
              data-replay-index="${index}" type="button"
              aria-pressed="${index === state.selectedReplay}">
        <span>0${index + 1}</span>
        <span><strong>${escapeHtml(replay.label)}</strong><small>${escapeHtml(replay.description)} · ${replay.timeline.length} 个事件</small></span>
      </button>`)
    .join("");
  document.querySelectorAll("[data-replay-index]").forEach((button) => {
    button.addEventListener("click", () => {
      selectReplay(Number(button.dataset.replayIndex));
    });
  });
}

function selectReplay(index) {
  stopReplay();
  state.selectedReplay = index;
  const replay = activeReplay();
  el("question").textContent = replay.run.question;
  el("question").title = replay.run.question;
  renderParity(replay);
  renderReplaySelector();
  renderReplayRecord(replay);
  resetReplay(false);
}

function resetTelemetry() {
  updateTelemetry({ tokens: 0, model_calls: 0, search_calls: 0, fetch_calls: 0, elapsed_seconds: 0 });
  el("answer-card").classList.add("pending");
  el("answer-status").textContent = "等待";
  el("answer-value").textContent = "等待轨迹重放。";
  el("raw-check").textContent = "原始答案 = 最终答案";
  el("score-check").textContent = "评分待显示";
}

function updateTelemetry(snapshot) {
  const limits = activeReplay().run.limits;
  const values = {
    tokens: snapshot.tokens || 0,
    model_calls: snapshot.model_calls || 0,
    search_calls: snapshot.search_calls || 0,
    fetch_calls: snapshot.fetch_calls || 0,
    elapsed_seconds: snapshot.elapsed_seconds || 0,
  };
  const tokenPct = ratioPercent(values.tokens, limits.tokens);
  el("token-value").textContent = values.tokens > 999 ? `${(values.tokens / 1000).toFixed(1)}k` : values.tokens;
  el("token-percent").textContent = `${tokenPct}%`;
  el("token-limit").textContent = `/ ${formatNumber(limits.tokens)}`;
  el("token-meter").style.width = `${tokenPct}%`;

  setBar("model", values.model_calls, limits.model_calls);
  setBar("search", values.search_calls, limits.search_calls);
  setBar("fetch", values.fetch_calls, limits.fetch_calls);
  el("wall-time").textContent = `${Number(values.elapsed_seconds).toFixed(1)} / ${limits.wall_time_seconds}s`;
  el("wall-bar").style.width = `${ratioPercent(values.elapsed_seconds, limits.wall_time_seconds)}%`;
}

function setBar(name, value, limit) {
  el(`${name}-calls`).textContent = `${value} / ${limit}`;
  el(`${name}-bar`).style.width = `${ratioPercent(value, limit)}%`;
}

function eventMeta(event) {
  if (event.kind === "search") return `${event.result_count} 个候选`;
  if (event.kind === "fetch") return `${formatNumber(event.content_chars)} 字符`;
  if (event.kind === "model" && event.budget) return `累计 ${formatNumber(event.budget.tokens)}`;
  return `#${String(event.sequence || 0).padStart(2, "0")}`;
}

function eventDetail(event) {
  if (event.kind === "search" && event.providers) {
    const statusLabels = {
      success: "成功",
      empty: "无结果",
      error: "失败",
      skipped: "已跳过",
      not_configured: "未配置",
    };
    const providerText = event.providers
      .map((item) => `${item.name}：${statusLabels[item.status] || item.status}`)
      .join("；");
    return `${event.detail} — ${providerText}`;
  }
  if (event.kind === "fetch") {
    if (event.status !== "success") return event.detail;
    const acquisitionLabels = {
      direct_http: "直接网页抓取",
      mediawiki_api: "MediaWiki 接口",
      provider_cache: "检索缓存",
    };
    const method = acquisitionLabels[event.acquisition_method] || "安全抓取";
    return `${event.detail}；来源 ${event.source_id || "未编号"}；${method}`;
  }
  return event.detail;
}

function eventStatus(event) {
  const labels = {
    success: "成功",
    completed: "已完成",
    logged: "已记录",
    warning: "需调整",
    recovered: "已恢复",
    stopped: "已停止",
  };
  return labels[event.status] || String(event.status || "已记录");
}

function addNextEvent() {
  const timeline = activeReplay().timeline;
  if (state.timelineIndex >= timeline.length) {
    finishReplay();
    return false;
  }
  const event = timeline[state.timelineIndex];
  document.querySelectorAll(".event-card.current").forEach((card) => card.classList.remove("current"));
  const card = document.createElement("article");
  const sourceLink = event.url
    ? `<a class="event-url" href="${escapeHtml(event.url)}" target="_blank" rel="noopener noreferrer">
        <span>抓取链接</span>${escapeHtml(event.url)}
      </a>`
    : "";
  card.className = "event-card current";
  card.dataset.kind = event.kind;
  card.dataset.status = event.status || "logged";
  card.innerHTML = `
    <span class="event-node">${icons[event.kind] || "·"}</span>
    <div class="event-copy">
      <strong>${escapeHtml(event.title)}</strong>
      <div class="event-origin">
        <span>模块 <code>${escapeHtml(event.module)}</code></span>
        <span>函数 <code>${escapeHtml(event.function)}</code></span>
      </div>
      <small title="${escapeHtml(eventDetail(event))}">${escapeHtml(eventDetail(event))}</small>
      ${sourceLink}
    </div>
    <div class="event-meta"><b>${escapeHtml(eventStatus(event))}</b>${escapeHtml(eventMeta(event))}</div>`;
  el("timeline-stage").appendChild(card);
  state.timelineIndex += 1;
  el("event-counter").textContent = `${state.timelineIndex} / ${timeline.length} 个事件`;
  if (event.budget) updateTelemetry(event.budget);
  if (event.kind === "search") {
    const current = telemetryFromRenderedEvents();
    current.elapsed_seconds += event.duration || 0;
    updateTelemetry(current);
  }
  if (event.kind === "fetch") {
    const current = telemetryFromRenderedEvents();
    current.elapsed_seconds += event.duration || 0;
    updateTelemetry(current);
  }
  el("timeline-stage").scrollTop = el("timeline-stage").scrollHeight;
  if (state.timelineIndex >= timeline.length && !state.replayTimer) finishReplay();
  return true;
}

function telemetryFromRenderedEvents() {
  const rendered = activeReplay().timeline.slice(0, state.timelineIndex);
  const modelEvents = rendered.filter((event) => event.kind === "model");
  const latestModel = modelEvents[modelEvents.length - 1];
  const latestBudget = latestModel && latestModel.budget ? latestModel.budget : {};
  return {
    tokens: latestBudget.tokens || 0,
    model_calls: modelEvents.length,
    search_calls: rendered.filter((event) => event.kind === "search").length,
    fetch_calls: rendered.filter((event) => event.kind === "fetch").length,
    elapsed_seconds: latestBudget.elapsed_seconds || 0,
  };
}

function startReplay() {
  stopReplay();
  resetReplay(false);
  el("timeline-stage").innerHTML = "";
  el("replay-button").classList.add("running");
  el("replay-button").lastChild.textContent = " 正在重放";
  state.replayTimer = window.setInterval(() => {
    if (!addNextEvent()) stopReplay();
  }, 520);
}

function stopReplay() {
  if (state.replayTimer) window.clearInterval(state.replayTimer);
  state.replayTimer = null;
  if (el("replay-button")) el("replay-button").classList.remove("running");
  if (el("replay-button")) el("replay-button").lastChild.textContent = " 重放执行轨迹";
}

function finishReplay() {
  stopReplay();
  const run = activeReplay().run;
  updateTelemetry({
    tokens: run.metrics.tokens,
    model_calls: run.metrics.model_calls,
    search_calls: run.metrics.search_calls,
    fetch_calls: run.metrics.fetch_calls,
    elapsed_seconds: run.metrics.wall_time_seconds,
  });
  const budgetStopped = run.completion_status === "budget_exhausted";
  el("answer-card").classList.remove("pending", "stopped");
  if (budgetStopped) el("answer-card").classList.add("stopped");
  el("answer-status").textContent = budgetStopped ? "预算停止" : "已完成";
  el("answer-value").textContent = run.final_answer || "未输出答案：预算保护已触发。";
  el("raw-check").textContent = budgetStopped
    ? "轨迹已保存"
    : run.answer_unchanged
      ? "原始答案 = 最终答案"
      : "答案发生变化";
  el("score-check").textContent = budgetStopped
    ? "无最终答案"
    : run.standard_em
      ? "标准 EM ✓"
      : "标准 EM ✕";
}

function resetReplay(stop = true) {
  if (stop) stopReplay();
  state.timelineIndex = 0;
  el("answer-card").classList.remove("stopped");
  el("timeline-stage").innerHTML = `
    <div class="timeline-empty">
      <span class="trace-icon">▶</span>
      <p>正式轨迹已加载</p>
      <small>点击“重放执行轨迹”查看 Agent 如何完成研究</small>
    </div>`;
  el("event-counter").textContent = `0 / ${activeReplay().timeline.length} 个事件`;
  resetTelemetry();
}

function setupControls() {
  el("replay-button").addEventListener("click", startReplay);
  el("step-button").addEventListener("click", () => {
    stopReplay();
    if (state.timelineIndex === 0) el("timeline-stage").innerHTML = "";
    addNextEvent();
  });
  el("reset-button").addEventListener("click", () => resetReplay(true));
}

async function init() {
  try {
    state.bundle = await loadBundle();
    renderHeader(state.bundle);
    renderReplaySelector();
    renderReplayRecord(activeReplay());
    setupControls();
    resetReplay(false);
  } catch (error) {
    document.body.innerHTML = `<main class="load-error"><h1>Harness Console failed to load</h1><pre>${escapeHtml(error.message)}</pre><p>Serve this directory over HTTP instead of opening index.html directly.</p></main>`;
    console.error(error);
  }
}

init();
