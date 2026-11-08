"use strict";
/* FedPEFT-IMU pipeline manager: single-page UI over /api (fedpeft/manager/server.py). */

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const isNum = (x) => typeof x === "number" && isFinite(x);
const f3 = (x) => (isNum(x) ? x.toFixed(3) : "–");
const sgn = (x) => (isNum(x) ? (x > 0 ? "+" : x < 0 ? "−" : "±") + Math.abs(x).toFixed(3) : "–");
function fmtMin(m) {
  if (!isNum(m)) return "–";
  if (m < 1) return "< 1 min";
  if (m < 60) return `${Math.round(m)} min`;
  const h = Math.floor(m / 60), r = Math.round(m % 60);
  return r ? `${h} h ${r} min` : `${h} h`;
}
const fmtSec = (s) => (isNum(s) ? fmtMin(s / 60) : "–");
const fmtTime = (t) => (t ? new Date(t * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "–");
const STATUS = { done: "Done", running: "Running", queued: "Queued", pending: "Queued", failed: "Failed", missing: "Not run", skipped: "Skipped", "n/a": "n/a" };
const chip = (st) => `<span class="st ${esc(st === "n/a" ? "na" : st)}">${esc(STATUS[st] || st)}</span>`;
const STAGES = [["prepare", "Prepare data"], ["weights", "Pretrained weights"], ["pretrain", "HALO pretraining"],
  ["centralized", "Centralized training"], ["federated", "Federated grid (FedAvg, V0–V5)"], ["published", "Published protocol"],
  ["ablation", "Ablation study"], ["trials", "Trials (extra user splits)"], ["report", "HTML report"]];
const STAGE_LABEL = Object.fromEntries(STAGES);

async function api(path, body) {
  const opt = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch(path, opt);
  let j;
  try { j = await r.json(); } catch { throw new Error(`${r.status} ${r.statusText}`); }
  if (!r.ok || (j && j.error)) throw new Error((j && j.error) || r.statusText);
  return j;
}
let toastTimer;
function toast(msg, bad = false) {
  const t = $("toast");
  t.textContent = msg; t.className = "toast" + (bad ? " bad" : ""); t.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => (t.hidden = true), bad ? 7000 : 3500);
}
async function act(fn, ok) {
  try { const r = await fn(); if (ok) toast(typeof ok === "function" ? ok(r) : ok); await tick(); return r; }
  catch (e) { toast(e.message, true); }
}

const S = { tab: "overview", ov: null, plDataset: null, abDataset: null, profile: null, profileSub: "keys",
  logKey: "runner", openStages: new Set(), mx: null, raInit: false, queue: null };

/* ---------------------------------------------------------------- tabs */
document.querySelectorAll("#tabs button").forEach((b) => (b.onclick = () => showTab(b.dataset.tab)));
function showTab(t) {
  S.tab = t;
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === t));
  document.querySelectorAll(".tab").forEach((s) => s.classList.toggle("active", s.id === "tab-" + t));
  refreshTab();
}
async function refreshTab() {
  try {
    if (S.tab === "pipeline") await loadPlan();
    else if (S.tab === "matrix") await loadMatrix();
    else if (S.tab === "ablations") await loadAblations();
    else if (S.tab === "settings") await loadProfiles();
    else if (S.tab === "queue") await loadQueue();
    else if (S.tab === "logs") await loadLog(true);
  } catch (e) { toast(e.message, true); }
}

/* ---------------------------------------------------------------- polling */
let ticking = false;
async function tick() {
  if (ticking) return;
  ticking = true;
  try {
    S.ov = await api("/api/overview");
    renderHeader();
    if (S.tab === "overview") renderOverview();
    if (S.tab === "queue") await loadQueue();
    if (S.tab === "logs") await loadLog(false);
  } catch (e) {
    $("runnerPill").className = "pill off"; $("runnerPill").textContent = "UI server unreachable";
  } finally { ticking = false; }
}
setInterval(tick, 2500);

/* ---------------------------------------------------------------- header */
function runnerState(ov) {
  const r = ov.runner, n = ov.counts;
  if (!r.alive) return n.pending ? ["off", "Not running", `${n.pending} job${n.pending > 1 ? "s" : ""} queued`] : ["off", "Idle", "the queue is empty"];
  const map = { running: ["running", "Running"], suspended: ["suspended", "Paused (job frozen)"], paused: ["paused", "Paused"],
    waiting: ["waiting", "Waiting"], idle: ["idle", "Idle"], finished: ["finished", "Finished"], starting: ["running", "Starting"], stopped: ["stopped", "Stopped"] };
  const [cls, label] = map[r.state] || ["off", r.state];
  return [cls, label, r.note || ""];
}
function renderHeader() {
  const ov = S.ov, r = ov.runner, c = ov.control, n = ov.counts;
  const [cls, label, note] = runnerState(ov);
  $("runnerPill").className = "pill " + cls; $("runnerPill").textContent = label;
  let extra = note;
  if (ov.current) extra = ov.current.item.job.label;
  if (c.after_current) extra += ` · will ${c.after_current} after this job`;
  if (n.pending) extra += ` · ${n.pending} queued (~${fmtMin(ov.queued_min)})`;
  $("runnerNote").textContent = extra;
  $("qCount").textContent = n.pending + n.running || "";
  const b = [];
  if (!r.alive || ["paused", "suspended", "stopped", "finished", "idle"].includes(r.state)) {
    if (n.pending || r.state === "suspended") b.push(`<button class="btn primary" data-ctl="start">▶ ${r.state === "suspended" || r.state === "paused" ? "Continue" : "Start queue"}</button>`);
  }
  if (r.alive && r.state === "running") {
    b.push(c.after_current === "pause" ? `<button class="btn" data-ctl="cancel_after">Don't pause after this job</button>`
      : `<button class="btn" data-ctl="pause_after" title="Finish the running job, then wait">⏸ Pause after this job</button>`);
    b.push(`<button class="btn" data-ctl="pause_now" title="Freeze the running job now; Continue resumes it exactly where it stopped">⏸ Pause now</button>`);
  }
  if (r.alive && r.state !== "stopped") b.push(`<button class="btn danger" data-ctl="stop_now" title="Stop the runner">■ Stop</button>`);
  $("controls").innerHTML = b.join("");
}
$("controls").addEventListener("click", (e) => {
  const a = e.target.closest("[data-ctl]");
  if (!a) return;
  const action = a.dataset.ctl;
  if (action === "stop_now" && S.ov.runner.child_pid &&
      !confirm("Stop now?\n\nThe running job is ended and goes back to the queue; it will restart from its beginning.\n(Use “Pause now” to keep its progress.)")) return;
  act(() => api("/api/control", { action }), { start: "Started", pause_after: "Will pause after this job", pause_now: "Paused",
    stop_now: "Stopped", cancel_after: "OK" }[action]);
});

/* ---------------------------------------------------------------- overview */
function renderOverview() {
  const ov = S.ov;
  const disk = isNum(ov.disk_free_gb) && ov.disk_free_gb < 15
    ? `<div class="alert ${ov.disk_free_gb < 5 ? "bad" : ""}">Only <b>${ov.disk_free_gb.toFixed(1)} GB</b> free on the disk of the results. Every ssl-wearables run saves a 43 MB model; below 5 GB the runner waits.</div>` : "";
  $("otherGpu").innerHTML = disk + (ov.other_gpu && ov.other_gpu.length
    ? `<div class="alert">Another training process is using the GPU (pid ${ov.other_gpu.map((p) => p.pid).join(", ")}). The runner waits until it ends (one GPU job at a time).<br><span class="mono">${esc(ov.other_gpu[0].cmd)}</span></div>` : "");
  renderNow(ov.current);
  renderRunAll();
  renderStagesTable();
}

function phaseBar(p) {
  const idx = p.phases.findIndex((x) => x.key === p.phase);
  return `<div class="phases">${p.phases.map((x, i) => `<div class="phase ${i < idx ? "past" : i === idx ? "cur" : ""}">${esc(x.label)}</div>`).join("")}</div>`;
}
function lineChart(series, { ymin, ymax, xlab = "" } = {}) {
  const W = 420, H = 150, L = 34, R = 8, T = 8, B = 20;
  const pts = series.flatMap((s) => s.points.filter((p) => isNum(p[1])));
  if (!pts.length) return `<svg viewBox="0 0 ${W} ${H}"><text x="${W / 2}" y="${H / 2}" text-anchor="middle" fill="var(--muted)" font-size="12">waiting for the first evaluation…</text></svg>`;
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs, x0 + 1);
  let y0 = ymin ?? Math.min(...ys), y1 = ymax ?? Math.max(...ys);
  if (y1 - y0 < 1e-6) { y0 -= 0.05; y1 += 0.05; }
  const sx = (x) => L + ((x - x0) / (x1 - x0)) * (W - L - R), sy = (y) => T + (1 - (y - y0) / (y1 - y0)) * (H - T - B);
  let g = "";
  for (let i = 0; i <= 3; i++) {
    const v = y0 + ((y1 - y0) * i) / 3, y = sy(v);
    g += `<line x1="${L}" x2="${W - R}" y1="${y}" y2="${y}" stroke="var(--line)"/><text x="${L - 4}" y="${y + 4}" text-anchor="end" font-size="10" fill="var(--muted)">${v.toFixed(2)}</text>`;
  }
  g += `<text x="${L}" y="${H - 5}" font-size="10" fill="var(--muted)">${x0}</text><text x="${W - R}" y="${H - 5}" font-size="10" fill="var(--muted)" text-anchor="end">${x1} ${esc(xlab)}</text>`;
  for (const s of series) {
    const p = s.points.filter((q) => isNum(q[1]));
    if (!p.length) continue;
    g += `<polyline fill="none" stroke="${s.color}" stroke-width="2" points="${p.map((q) => `${sx(q[0]).toFixed(1)},${sy(q[1]).toFixed(1)}`).join(" ")}"/>`;
    const last = p[p.length - 1];
    g += `<circle cx="${sx(last[0])}" cy="${sy(last[1])}" r="3" fill="${s.color}"/>`;
  }
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${g}</svg>`;
}
const legend = (items) => `<span class="legend">${items.map(([c, n]) => `<span><i style="background:${c}"></i>${esc(n)}</span>`).join(" &nbsp;")}</span>`;

let nowLogKey = null, nowLogText = "";
async function renderNow(cur) {
  if (!cur) {
    const r = S.ov.runner;
    $("nowMeta").textContent = "";
    $("nowBody").innerHTML = `<p class="muted">${r.alive && r.state === "paused" ? "Paused between jobs. Press Continue to go on." :
      S.ov.counts.pending ? "Nothing is running. Press “Start queue” to run the queued jobs." : "Nothing is running and the queue is empty."}</p>`;
    return;
  }
  const it = cur.item, j = it.job, p = cur.progress;
  $("nowMeta").textContent = `${j.dataset} · ${STAGE_LABEL[j.stage] || j.stage}${it.attempts > 1 ? ` · attempt ${it.attempts}` : ""}`;
  const pct = p.total ? Math.min(100, (100 * p.step) / p.total) : null;
  const last = p.curve.length ? p.curve[p.curve.length - 1] : null;
  const fin = p.final;
  const finalBox = fin ? `<div class="final">
      <div><small>Macro-F1 (test)</small><b>${f3(fin.macro_f1)}</b></div><div><small>Accuracy</small><b>${f3(fin.accuracy)}</b></div>
      <div><small>Client accuracy</small><b>${f3(fin.client_acc)}</b></div><div><small>Worst-10 %</small><b>${f3(fin.worst10)}</b></div>
      ${isNum(fin.zero_shot) ? `<div><small>New users, zero-shot</small><b>${f3(fin.zero_shot)}</b></div>` : ""}
      ${isNum(fin.uplink_mb) ? `<div><small>Uplink / client·round</small><b>${fin.uplink_mb.toFixed(3)} MB</b></div>` : ""}</div>` : "";
  const isFl = p.kind === "fl";
  const xs = p.curve.map((c) => c.x);
  const charts = p.kind === "other" ? "" : `<div class="charts">
      <div class="chart"><div class="ct">Training loss ${legend([["var(--wait)", p.kind === "halo" ? "MAE" : "loss"]])}</div>
        ${lineChart([{ color: "var(--wait)", points: p.curve.map((c) => [c.x, c.loss]) }], { xlab: p.unit + "s" })}</div>
      <div class="chart"><div class="ct">${isFl ? "Evaluation (clients' test windows)" : p.kind === "halo" ? "Contrastive loss" : "Validation macro-F1"}
        ${legend(isFl ? [["var(--accent)", "macro-F1"], ["var(--ok)", "accuracy"], ["var(--skip)", "worst-10 %"]] : [["var(--accent)", p.kind === "halo" ? "InfoNCE" : "val-F1"]])}</div>
        ${lineChart(isFl ? [{ color: "var(--accent)", points: p.curve.map((c) => [c.x, c.f1]) }, { color: "var(--ok)", points: p.curve.map((c) => [c.x, c.acc]) },
          { color: "var(--skip)", points: p.curve.map((c) => [c.x, c.worst10]) }]
          : [{ color: "var(--accent)", points: p.curve.map((c) => [c.x, p.kind === "halo" ? c.contrastive : c.f1]) }], { xlab: p.unit + "s" })}</div></div>`;
  $("nowBody").innerHTML = `
    <h3>${esc(j.label)}</h3>
    <div class="muted small mono">${esc("python scripts/" + j.script + " " + j.args.join(" "))}</div>
    ${phaseBar(p)}
    ${pct !== null ? `<div class="progress"><span style="width:${pct.toFixed(1)}%"></span></div>` : ""}
    <div class="kv">
      ${p.total ? `<span>${esc(p.unit)} <b>${p.step} / ${p.total}</b></span>` : ""}
      <span>elapsed <b>${fmtSec(cur.elapsed_s)}</b></span>
      <span>${p.phase === "train" ? "left" : "expected left"} <b>${fmtSec(cur.eta_s)}</b></span>
      ${last && isNum(last.s_per_round) ? `<span><b>${last.s_per_round.toFixed(1)} s</b> / round</span>` : ""}
      ${p.info && p.info.clients ? `<span><b>${p.info.clients}</b> clients, <b>${p.info.per_round}</b> per round</span>` : ""}
      ${last && isNum(last.f1) ? `<span>latest ${isFl ? "F1" : "val-F1"} <b>${f3(last.f1)}</b></span>` : ""}
    </div>
    ${finalBox}${charts}
    ${p.error ? `<div class="alert bad"><b>Error</b><pre class="log short">${esc(p.error)}</pre></div>` : ""}
    <pre class="log short" id="nowLog">${esc(nowLogKey === it.key ? nowLogText : "")}</pre>
    <div class="toolbar" style="margin-top:10px"><button class="btn sm" data-openlog="${esc(it.key)}">Full log</button>
      <button class="btn sm danger" data-skip="${esc(it.key)}">Skip this job</button></div>`;
  try {
    const lg = await api(`/api/log?key=${encodeURIComponent(it.key)}&lines=14`);
    nowLogKey = it.key; nowLogText = lg.text;
    const el = $("nowLog"); if (el) el.textContent = lg.text;
  } catch { /* the job may just have ended */ }
}
$("nowBody").addEventListener("click", (e) => {
  const o = e.target.closest("[data-openlog]");
  if (o) { S.logKey = o.dataset.openlog; showTab("logs"); return; }
  const s = e.target.closest("[data-skip]");
  if (s && confirm("Skip the running job? It is ended and marked skipped (no result).")) act(() => api("/api/control", { action: "skip_current" }), "Skipped");
});

/* ---------- run everything ---------- */
function renderRunAll() {
  const ov = S.ov;
  if (!S.raInit) {
    $("raDatasets").innerHTML = ov.datasets.map((d) => `<label><input type="checkbox" value="${esc(d.key)}" checked> ${esc(d.name)}</label>`).join("");
    $("raStages").innerHTML = STAGES.map(([k, l]) => `<label><input type="checkbox" value="${k}" checked> ${esc(l)}</label>`).join("");
    $("raFrom").innerHTML = STAGES.map(([k, l]) => `<option value="${k}">${esc(l)}</option>`).join("");
    for (const id of ["raDatasets", "raStages", "raFrom"]) $(id).addEventListener("change", renderRunAll);
    S.raInit = true;
  }
  const ds = [...$("raDatasets").querySelectorAll("input:checked")].map((i) => i.value);
  const st = [...$("raStages").querySelectorAll("input:checked")].map((i) => i.value);
  const from = STAGES.findIndex(([k]) => k === $("raFrom").value);
  let mins = 0, jobs = 0;
  for (const d of ov.datasets) {
    if (!ds.includes(d.key)) continue;
    for (const s of d.stages) {
      const i = STAGES.findIndex(([k]) => k === s.stage);
      if (i < from || !st.includes(s.stage)) continue;
      mins += s.missing_min; jobs += s.total - s.done;
    }
  }
  $("raEstimate").textContent = jobs ? `~${fmtMin(mins)}` : "Nothing to run";
  const nCal = Object.keys(ov.estimate_factors || {}).length;
  const cal = nCal ? `, corrected per dataset on ${ov.estimate_jobs} finished jobs` : "";
  $("raJobs").textContent = jobs ? `${jobs} job${jobs > 1 ? "s" : ""} without a result (estimate from earlier runs of the same kind${cal})` : "every selected job has a result";
  $("raGo").disabled = !jobs;
}
$("raGo").onclick = () => {
  const datasets = [...$("raDatasets").querySelectorAll("input:checked")].map((i) => i.value);
  const stages = [...$("raStages").querySelectorAll("input:checked")].map((i) => i.value);
  act(() => api("/api/enqueue", { datasets, stages, from_stage: $("raFrom").value, start: true }),
    (r) => `Queued ${r.added} job${r.added === 1 ? "" : "s"}; the runner ${r.runner && r.runner.started ? "started" : "is running"}`);
};

function renderStagesTable() {
  const ov = S.ov;
  $("missingTotal").textContent = `work without a result: ~${fmtMin(ov.missing_min)}`;
  const cols = STAGES.filter(([k]) => ov.datasets.some((d) => d.stages.some((s) => s.stage === k)));
  const head = `<tr><th>Dataset</th>${cols.map(([, l]) => `<th>${esc(l)}</th>`).join("")}</tr>`;
  const rows = ov.datasets.map((d) => `<tr><td><b>${esc(d.name)}</b><div class="muted small">profile ${esc(d.profile)}</div></td>${cols.map(([k]) => {
    const s = d.stages.find((x) => x.stage === k);
    if (!s) return `<td class="muted">–</td>`;
    const w = (n) => ((100 * n) / s.total).toFixed(1);
    return `<td><div class="stage-cell" data-ds="${esc(d.key)}" data-stage="${k}"><span>${s.done} / ${s.total}${s.failed ? ` · <span style="color:var(--bad)">${s.failed} failed</span>` : ""}</span>
      <div class="bar"><span class="d" style="width:${w(s.done)}%"></span><span class="r" style="width:${w(s.running)}%"></span><span class="q" style="width:${w(s.queued)}%"></span><span class="f" style="width:${w(s.failed)}%"></span></div>
      <small>${s.done === s.total ? "complete" : `~${fmtMin(s.missing_min)} left${s.queued ? `, ${s.queued} queued` : ""}`}</small></div></td>`;
  }).join("")}</tr>`).join("");
  $("stagesTable").innerHTML = head + rows;
}
$("stagesTable").addEventListener("click", (e) => {
  const c = e.target.closest("[data-stage]");
  if (!c) return;
  S.plDataset = c.dataset.ds; S.openStages = new Set([c.dataset.stage]);
  showTab("pipeline");
});

/* ---------------------------------------------------------------- queue */
async function loadQueue() {
  S.queue = await api("/api/queue");
  const q = S.queue, f = $("qFilter").value;
  const c = q.counts;
  $("qSummary").textContent = `${c.pending} queued · ${c.running} running · ${c.done} done · ${c.failed} failed · ${c.skipped} skipped`;
  const items = q.items.filter((it) => !f || it.status === f);
  if (!items.length) { $("queueTable").innerHTML = `<tr><td class="muted">${q.items.length ? "No job with this status." : "The queue is empty. Use “Run the full process” on the Overview, or queue jobs from the Pipeline, Encoders or Ablations tabs."}</td></tr>`; return; }
  let pendingIdx = 0;
  $("queueTable").innerHTML = `<tr><th>#</th><th>Status</th><th>Job</th><th>Dataset</th><th>Stage</th><th class="num">Expected</th><th>Started</th><th class="num">Took</th><th></th></tr>` +
    items.map((it) => {
      const took = it.started && it.ended ? fmtSec(it.ended - it.started) : it.status === "running" && it.started ? fmtSec(Date.now() / 1000 - it.started) : "–";
      const pos = it.status === "pending" ? ++pendingIdx : "";
      const acts = [];
      if (it.status === "pending") acts.push(`<button class="btn sm" data-i="start_from" title="Skip every queued job above this one and run from here">▶ Start from here</button>`,
        `<button class="btn sm" data-i="top" title="Run next">⤒</button>`, `<button class="btn sm" data-i="up">↑</button>`, `<button class="btn sm" data-i="down">↓</button>`,
        `<button class="btn sm" data-i="skip">Skip</button>`, `<button class="btn sm danger" data-i="remove">✕</button>`);
      if (it.status === "failed" || it.status === "skipped") acts.push(`<button class="btn sm" data-i="retry">Retry</button>`, `<button class="btn sm" data-i="start_from" title="Run again from this job on">▶ Start from here</button>`, `<button class="btn sm danger" data-i="remove">✕</button>`);
      if (it.status === "done") acts.push(`<button class="btn sm" data-i="rerun" title="Run again and overwrite the result">Re-run</button>`, `<button class="btn sm danger" data-i="remove">✕</button>`);
      return `<tr data-key="${esc(it.key)}"><td class="muted">${pos}</td><td>${chip(it.status)}</td>
        <td class="label"><a href="#" data-detail>${esc(it.label)}</a>${it.rerun ? ` <span class="muted small">(re-run)</span>` : ""}${it.note ? `<div class="muted small">${esc(it.note)}</div>` : ""}</td>
        <td>${esc(it.dataset)}</td><td class="small">${esc(STAGE_LABEL[it.stage] || it.stage)}</td><td class="num">${fmtMin(isNum(it.est_min) ? it.est_min * ((S.ov && (S.ov.estimate_factors || {})[it.dataset]) || 1) : null)}</td>
        <td class="small">${fmtTime(it.started)}</td><td class="num">${took}</td><td class="acts">${acts.join(" ")}</td></tr>`;
    }).join("");
}
$("qFilter").onchange = () => loadQueue();
document.querySelectorAll("[data-clear]").forEach((b) => (b.onclick = () => {
  if (b.dataset.clear === "pending" && !confirm("Remove every queued (not yet started) job from the queue?")) return;
  act(() => api("/api/clear", { what: b.dataset.clear }), (r) => `Removed ${r.removed}`);
}));
$("queueTable").addEventListener("click", (e) => {
  const tr = e.target.closest("tr[data-key]");
  if (!tr) return;
  const key = tr.dataset.key;
  if (e.target.closest("[data-detail]")) { e.preventDefault(); openItem(key); return; }
  const b = e.target.closest("[data-i]");
  if (!b) return;
  const action = b.dataset.i;
  if (action === "rerun" && !confirm("Run this job again? Its result file is overwritten.")) return;
  act(() => api("/api/item", { key, action }), (r) => action === "start_from" ? `Starting here (${r.skipped} queued job${r.skipped === 1 ? "" : "s"} above skipped)` : "OK");
});
async function openItem(key) {
  const it = (S.queue || { items: [] }).items.find((x) => x.key === key);
  if (!it) return;
  let log = "";
  try { log = (await api(`/api/log?key=${encodeURIComponent(key)}&lines=300`)).text; } catch { log = "(no log yet)"; }
  drawer(it.label, `<dl class="props"><dt>Status</dt><dd>${chip(it.status)} ${it.rc !== null && it.rc !== undefined ? `exit code ${it.rc}` : ""}</dd>
    <dt>Dataset · stage</dt><dd>${esc(it.dataset)} · ${esc(STAGE_LABEL[it.stage] || it.stage)} (profile ${esc(it.profile)})</dd>
    <dt>Command</dt><dd class="mono">${esc(it.command)}${it.rerun ? " --rerun" : ""}</dd>
    <dt>Added · started · ended</dt><dd>${fmtTime(it.added)} · ${fmtTime(it.started)} · ${fmtTime(it.ended)}</dd>
    <dt>Attempts</dt><dd>${it.attempts}</dd>${it.note ? `<dt>Note</dt><dd>${esc(it.note)}</dd>` : ""}</dl>
    ${it.error ? `<h3>Error</h3><pre class="log">${esc(it.error)}</pre>` : ""}<h3>Log</h3><pre class="log">${esc(log)}</pre>`);
}

/* ---------------------------------------------------------------- pipeline */
function dsSeg(el, current, onPick) {
  const ds = S.ov ? S.ov.datasets : [];
  el.innerHTML = ds.map((d) => `<button data-ds="${esc(d.key)}" class="${d.key === current ? "active" : ""}">${esc(d.name)}</button>`).join("");
  el.onclick = (e) => { const b = e.target.closest("[data-ds]"); if (b) onPick(b.dataset.ds); };
}
async function ensureOv() { if (!S.ov) S.ov = await api("/api/overview"); }
async function loadPlan() {
  await ensureOv();
  S.plDataset = S.plDataset || S.ov.order[0];
  dsSeg($("plDatasets"), S.plDataset, (d) => { S.plDataset = d; S.openStages = new Set(); loadPlan(); });
  const pl = await api(`/api/plan?dataset=${encodeURIComponent(S.plDataset)}`);
  $("plStages").innerHTML = pl.stages.map((st, i) => {
    const n = st.jobs.length, done = st.jobs.filter((j) => j.status === "done").length;
    const miss = st.jobs.filter((j) => j.status !== "done").reduce((a, j) => a + (j.est_min || 10), 0);
    const open = S.openStages.has(st.stage);
    const cnt = (s) => st.jobs.filter((j) => j.status === s).length;
    const w = (k) => ((100 * cnt(k)) / n).toFixed(1);
    return `<div class="card stage ${done === n ? "complete" : ""}" data-stage="${st.stage}">
      <div class="card-head" data-toggle><div class="stage-title"><span class="num-badge">${i + 1}</span><div><h2>${esc(st.label)}</h2>
        <div class="muted small">${done} / ${n} done${done < n ? ` · ~${fmtMin(miss)} left` : ""}${cnt("queued") ? ` · ${cnt("queued")} queued` : ""}${cnt("failed") ? ` · ${cnt("failed")} failed` : ""}</div></div></div>
        <div class="toolbar"><span class="mini-bar"><span style="width:${w("done")}%;background:var(--ok)"></span><span style="width:${w("running")}%;background:var(--run)"></span><span style="width:${w("queued")}%;background:var(--wait)"></span><span style="width:${w("failed")}%;background:var(--bad)"></span></span>
          ${done < n ? `<button class="btn sm" data-q="stage">Queue missing</button><button class="btn sm primary" data-q="from" title="Queue this stage and every later stage of this dataset, then run">▶ Start from this stage</button>` : ""}
          <button class="btn sm" data-toggle>${open ? "Hide jobs" : "Show jobs"}</button></div></div>
      ${open ? `<div class="jobs"><table class="list">${jobRows(st.jobs)}</table></div>` : ""}</div>`;
  }).join("");
}
function jobRows(jobs) {
  const head = `<tr><th>Status</th><th>Job</th><th>Group</th><th class="num">Macro-F1</th><th class="num">+FT</th><th class="num">New users</th><th class="num">Expected</th><th></th></tr>`;
  return head + jobs.map((j) => {
    const r = j.result || {};
    const acts = j.status === "done" ? `<button class="btn sm" data-job="${esc(j.id)}" data-rerun="1">Re-run</button>`
      : j.status === "running" || j.status === "queued" ? "" : `<button class="btn sm" data-job="${esc(j.id)}">Queue</button>`;
    return `<tr title="${esc(j.command)}"><td>${chip(j.status)}</td><td class="label">${esc(j.label)}</td><td class="small muted">${esc(j.group)}</td>
      <td class="num">${f3(r.macro_f1)}</td><td class="num">${f3(r.ft_macro_f1)}</td><td class="num">${f3(r.zero_shot_f1)}</td><td class="num">${fmtMin(j.est_min)}</td><td class="acts">${acts}</td></tr>`;
  }).join("");
}
$("plStages").addEventListener("click", (e) => {
  const card = e.target.closest("[data-stage]");
  if (!card) return;
  const stage = card.dataset.stage;
  const job = e.target.closest("[data-job]");
  if (job) { queueJob(job.dataset.job, !!job.dataset.rerun); return; }
  const q = e.target.closest("[data-q]");
  if (q) {
    const body = q.dataset.q === "stage" ? { datasets: [S.plDataset], stages: [stage] } : { datasets: [S.plDataset], from_stage: stage, start: true };
    act(() => api("/api/enqueue", body), (r) => `Queued ${r.added} job${r.added === 1 ? "" : "s"}${body.start ? "; running" : ""}`).then(loadPlan);
    return;
  }
  if (e.target.closest("[data-toggle]")) {
    S.openStages.has(stage) ? S.openStages.delete(stage) : S.openStages.add(stage);
    loadPlan();
  }
});
function queueJob(id, rerun) {
  if (rerun && !confirm("Run this job again? Its result file is overwritten.")) return;
  act(() => api("/api/enqueue", { ids: [id], rerun }), (r) => (r.added ? "Queued (start the queue to run it)" : "Already queued")).then(refreshTab);
}

/* ---------------------------------------------------------------- matrix */
const MX_FMT = { uplink_mb: (x) => (isNum(x) ? (x < 0.1 ? x.toFixed(3) : x.toFixed(2)) : "–"), minutes: (x) => (isNum(x) ? x.toFixed(1) : "–") };
async function loadMatrix() {
  S.mx = await api("/api/matrix");
  renderMatrix();
}
$("mxMetric").onchange = () => renderMatrix();
function renderMatrix() {
  const mx = S.mx, metric = $("mxMetric").value, fmt = MX_FMT[metric] || f3;
  const lowerBetter = metric === "uplink_mb" || metric === "minutes";
  let html = `<tr><th class="left">Encoder · method</th>${mx.datasets.map((d) => `<th>${esc(d.name)}</th>`).join("")}</tr>`;
  for (const enc of mx.encoders) {
    const rows = mx.methods.filter((m) => mx.datasets.some((d) => mx.cells[`${d.key}|${enc.key}|${m.key}`]));
    if (!rows.length) continue;
    const best = {};
    for (const d of mx.datasets) {
      const vals = rows.filter((m) => m.kind === "federated").map((m) => (mx.cells[`${d.key}|${enc.key}|${m.key}`] || {})[metric]).filter(isNum);
      if (vals.length) best[d.key] = lowerBetter ? Math.min(...vals) : Math.max(...vals);
    }
    html += `<tr><td class="enc" colspan="${mx.datasets.length + 1}">${esc(enc.name)}</td></tr>`;
    for (const m of rows) {
      html += `<tr><td class="left">${esc(m.short)}</td>${mx.datasets.map((d) => {
        const c = mx.cells[`${d.key}|${enc.key}|${m.key}`];
        if (!c) return `<td class="muted">–</td>`;
        const v = c[metric];
        if (!isNum(v)) return `<td class="cell" data-cell="${esc(d.key)}|${esc(enc.key)}|${esc(m.key)}">${c.status === "done" ? "–" : chip(c.status)}</td>`;
        const isBest = m.kind === "federated" && best[d.key] === v;
        return `<td class="cell ${isBest ? "best" : ""}" data-cell="${esc(d.key)}|${esc(enc.key)}|${esc(m.key)}" title="${c.seeds.length} split(s): ${c.seeds.join(", ")}">${fmt(v)}${c.seeds.length > 1 ? `<sup>${c.seeds.length}</sup>` : ""}</td>`;
      }).join("")}</tr>`;
    }
  }
  $("matrixTable").innerHTML = html;
}
$("matrixTable").addEventListener("click", (e) => {
  const td = e.target.closest("[data-cell]");
  if (!td) return;
  const [ds, enc, m] = td.dataset.cell.split("|");
  const c = S.mx.cells[td.dataset.cell];
  const d = S.mx.datasets.find((x) => x.key === ds);
  const central = m === "scratch" || m === "pretrained_ft";
  const id = central ? `central/${d.profile}/${enc}/${m}/s0` : `fl/${d.profile}/${enc}/${m}/s0`;
  const s0 = c.seed0 || {};
  drawer(`${(S.mx.encoders.find((x) => x.key === enc) || {}).name} · ${(S.mx.methods.find((x) => x.key === m) || {}).short} · ${d.name}`,
    `<dl class="props"><dt>Status (split 0)</dt><dd>${chip(c.status)}</dd><dt>Splits with a result</dt><dd>${c.seeds.join(", ") || "none"}</dd>
     <dt>Macro-F1</dt><dd>${f3(c.macro_f1)} (split 0: ${f3(s0.macro_f1)})</dd><dt>+FT macro-F1</dt><dd>${f3(c.ft_macro_f1)}</dd>
     <dt>Client accuracy</dt><dd>${f3(c.client_acc)}</dd><dt>Worst-10 % F1</dt><dd>${f3(c.worst10_f1)}</dd>
     <dt>New users, zero-shot</dt><dd>${f3(c.zero_shot_f1)}</dd><dt>Uplink</dt><dd>${MX_FMT.uplink_mb(c.uplink_mb)} MB / client / round</dd>
     <dt>Run time</dt><dd>${fmtMin(c.minutes)}</dd><dt>Result file</dt><dd class="mono">${esc(s0.file || "–")}</dd></dl>
     <div class="toolbar">${c.status === "done" ? `<button class="btn" data-dq="${esc(id)}" data-rerun="1">Re-run split 0 (overwrite)</button>`
       : c.status === "missing" || c.status === "failed" ? `<button class="btn primary" data-dq="${esc(id)}">Queue split 0</button>` : ""}</div>`);
});

/* ---------------------------------------------------------------- ablations */
const GROUPS = { reference: "Reference", rank: "LoRA rank (V3, exact aggregation)", components: "V5 without one component",
  switches: "V5 with a switch", grid: "Read from the main grid", other: "Other tagged runs (earlier experiments)" };
async function loadAblations() {
  await ensureOv();
  S.abDataset = S.abDataset || S.ov.order[0];
  dsSeg($("abDatasets"), S.abDataset, (d) => { S.abDataset = d; loadAblations(); });
  const ab = await api(`/api/ablations?dataset=${encodeURIComponent(S.abDataset)}`);
  $("abTitle").textContent = `Ablation study · ${ab.name}`;
  const def = ab.definition || {};
  $("abLede").innerHTML = `Encoders: ${esc((def.encoders || []).join(", "))} · user split ${esc((def.seeds || [0]).join(", "))} · profile <span class="mono">${esc(ab.profile)}</span>.
    Δ is the macro-F1 difference to V5 (rank rows: to the main V3 run). Shared-classifier methods are also shown after +FT, where each client fine-tunes the shared classifier.
    The study is defined in <span class="mono">configs/manager.yaml</span> (Settings → Study definition).`;
  $("abTables").innerHTML = ab.encoders.map((e) => {
    let g = null, rows = "";
    for (const r of e.rows) {
      if (r.group !== g) { g = r.group; rows += `<tr class="grp"><td colspan="10">${esc(GROUPS[g] || g)}</td></tr>`; }
      const x = r.result || {};
      const acts = r.job_id ? (r.status === "done" ? `<button class="btn sm" data-job="${esc(r.job_id)}" data-rerun="1">Re-run</button>`
        : r.status === "missing" || r.status === "failed" ? `<button class="btn sm" data-job="${esc(r.job_id)}">Queue</button>` : "") : "";
      rows += `<tr><td class="label">${esc(r.label)}${r.note ? ` <span class="muted small">${esc(r.note)}</span>` : ""}</td><td>${chip(r.status)}</td>
        <td class="num">${f3(x.macro_f1)}</td><td class="num">${f3(x.ft_macro_f1)}</td>
        <td class="num"><span class="delta ${isNum(r.delta_f1) ? (r.delta_f1 < -0.0005 ? "neg" : r.delta_f1 > 0.0005 ? "pos" : "") : ""}">${sgn(r.delta_f1)}</span></td>
        <td class="num">${f3(x.client_acc)}</td><td class="num">${f3(x.worst10_f1)}</td><td class="num">${f3(x.zero_shot_f1)}</td>
        <td class="num">${MX_FMT.uplink_mb(x.uplink_mb)}</td><td class="acts">${acts}</td></tr>`;
    }
    const miss = e.rows.filter((r) => r.job_id && r.status === "missing").length;
    return `<div class="ab-enc"><h3>${esc(e.name)} ${miss ? `<span class="muted small">· ${miss} missing</span>` : `<span class="st done">complete</span>`}</h3>
      <div class="table-wrap"><table class="list"><tr><th>Variant</th><th>Status</th><th class="num">Macro-F1</th><th class="num">+FT</th><th class="num">Δ F1</th>
      <th class="num">Client acc</th><th class="num">Worst-10 %</th><th class="num">New users</th><th class="num">Uplink MB</th><th></th></tr>${rows}</table></div></div>`;
  }).join("") || `<p class="muted">No ablation encoder of this dataset.</p>`;
}
$("abTables").addEventListener("click", (e) => { const b = e.target.closest("[data-job]"); if (b) queueJob(b.dataset.job, !!b.dataset.rerun); });
$("abQueueDs").onclick = () => act(() => api("/api/enqueue", { datasets: [S.abDataset], stages: ["ablation"] }), (r) => `Queued ${r.added} ablation run${r.added === 1 ? "" : "s"} (press Start queue)`).then(loadAblations);
$("abQueueAll").onclick = () => act(() => api("/api/enqueue", { stages: ["ablation"], start: true }), (r) => `Queued ${r.added} ablation run${r.added === 1 ? "" : "s"}; running`).then(loadAblations);

/* ---------------------------------------------------------------- settings */
async function loadProfiles() {
  await ensureOv();
  const pr = await api("/api/profiles");
  const names = Object.fromEntries(S.ov.datasets.map((d) => [d.key, d.name]));
  const groups = {};
  for (const p of pr.profiles) {
    const g = p.manager_dataset ? names[p.manager_dataset] : `Other (${p.dataset})`;
    (groups[g] = groups[g] || []).push(p);
  }
  const order = [...S.ov.datasets.map((d) => d.name), ...Object.keys(groups).filter((g) => !Object.values(names).includes(g)).sort()];
  $("profileList").innerHTML = `<div class="grp-title">Full process</div><button data-p="__manager__" class="${S.profile === "__manager__" ? "active" : ""}">Study definition<span class="role">manager.yaml</span></button>` +
    order.filter((g) => groups[g]).map((g) => `<div class="grp-title">${esc(g)}</div>` + groups[g].sort((a, b) => (a.role === "main" ? -1 : b.role === "main" ? 1 : 0))
      .map((p) => `<button data-p="${esc(p.name)}" class="${p.name === S.profile ? "active" : ""}" title="${esc(p.description)}">${esc(p.name)}<span class="role">${p.role === "main" ? "main" : p.role === "published" ? "published protocol" : ""}</span></button>`).join("")).join("");
  if (S.profile) await loadProfile(S.profile);
}
$("profileList").addEventListener("click", (e) => { const b = e.target.closest("[data-p]"); if (b) { S.profile = b.dataset.p; S.profileSub = "keys"; loadProfiles(); } });

async function loadProfile(name) {
  if (name === "__manager__") {
    const m = await api("/api/manager_config");
    $("profileDetail").innerHTML = `<div class="card"><div class="card-head"><h2>Study definition</h2><span class="mono muted">configs/manager.yaml</span></div>
      <p class="muted small">What “the full process” runs for each dataset: the stages, the published protocols, the ablation study and the extra user splits (trials). The training settings themselves are in the profiles. A backup of the old file is kept in logs/manager/backups/.</p>
      <textarea id="mgrText" spellcheck="false">${esc(m.text)}</textarea>
      <div class="toolbar" style="margin-top:10px"><button class="btn primary" id="mgrSave">Save</button></div></div>`;
    $("mgrSave").onclick = () => act(() => api("/api/profile/save", { kind: "manager", text: $("mgrText").value }), "Saved; the plan is updated");
    return;
  }
  const d = await api(`/api/profile?name=${encodeURIComponent(name)}`);
  const sub = S.profileSub;
  const subtabs = [["keys", "Key settings"], ["profile", `Profile file`], ["dataset", `Dataset file`], ["budgets", "Encoder budgets"], ["effective", "Effective configuration"]];
  let body = "";
  if (sub === "keys") {
    body = `<p class="muted small">The value each run uses, and where it comes from: <span class="src">default</span> configs/default.yaml, <span class="src dataset">dataset</span> configs/datasets/${esc(d.dataset)}.yaml,
      <span class="src profile">profile</span> this profile. Saving writes the value into the profile (comments are kept). Changes apply to jobs that start afterwards; existing results stay until you re-run them.</p>` +
      d.groups.map((g) => `<h3>${esc(g.title)}</h3>` + g.settings.map((s) => settingRow(s)).join("")).join("");
  } else if (sub === "profile") {
    body = `<p class="muted small">configs/profiles/${esc(name)}.yaml: overrides the dataset file and default.yaml. The file is checked (it must load) before it is saved.</p>
      <textarea id="pText" spellcheck="false">${esc(d.profile_text)}</textarea><div class="toolbar" style="margin-top:10px"><button class="btn primary" data-save="profile">Save profile</button></div>`;
  } else if (sub === "dataset") {
    body = d.dataset_file ? `<p class="muted small">configs/datasets/${esc(d.dataset_file)}: shared by every profile of this dataset.</p>
      <textarea id="dText" spellcheck="false">${esc(d.dataset_text)}</textarea><div class="toolbar" style="margin-top:10px"><button class="btn primary" data-save="dataset">Save dataset file</button></div>`
      : `<p class="muted">This dataset has no dataset file (its settings are in default.yaml).</p>`;
  } else if (sub === "budgets") {
    body = `<p class="muted small">Some encoders run with their own budget (encoders.&lt;name&gt;.budget), e.g. fewer rounds for the slow UniMTS. These are the values each encoder actually uses.</p>
      <table class="list"><tr><th>Encoder</th><th class="num">FL rounds</th><th class="num">Local steps</th><th class="num">Batch</th><th class="num">Central epochs</th><th>Methods</th></tr>
      ${d.budgets.map((b) => `<tr><td>${esc(b.encoder)}${b.budget ? ` <span class="src dataset">budget</span>` : ""}</td><td class="num">${b.rounds}</td><td class="num">${b.max_local_steps}</td><td class="num">${b.batch_size}</td><td class="num">${b.centralized_epochs}</td><td class="small">${esc(b.methods.join(", "))}</td></tr>`).join("")}</table>`;
  } else {
    body = `<pre class="log">${esc(d.effective)}</pre>`;
  }
  $("profileDetail").innerHTML = `<div class="card"><div class="card-head"><div><h2>${esc(name)}</h2><div class="muted small">dataset ${esc(d.dataset)}</div></div>
      <div class="toolbar"><button class="btn" id="pDup">Duplicate…</button></div></div>
    <div class="subtabs">${subtabs.map(([k, l]) => `<button data-sub="${k}" class="${k === sub ? "active" : ""}">${esc(l)}</button>`).join("")}</div>${body}</div>`;
  $("profileDetail").querySelector(".subtabs").onclick = (e) => { const b = e.target.closest("[data-sub]"); if (b) { S.profileSub = b.dataset.sub; loadProfile(name); } };
  $("pDup").onclick = () => {
    const nn = prompt("Name of the new profile (lower-case letters, digits, _):", name + "_copy");
    if (nn) act(() => api("/api/profile/duplicate", { name, new: nn }), "Profile created").then(() => { S.profile = nn; loadProfiles(); });
  };
  $("profileDetail").querySelectorAll("[data-save]").forEach((b) => (b.onclick = () => {
    const kind = b.dataset.save;
    const text = kind === "profile" ? $("pText").value : $("dText").value;
    const target = kind === "profile" ? name : d.dataset_file.replace(/\.yaml$/, "");
    act(() => api("/api/profile/save", { kind, name: target, text }), (r) => `Saved ${r.saved}${r.backup ? ` (backup ${r.backup})` : ""}`).then(() => loadProfile(name));
  }));
  $("profileDetail").querySelectorAll(".setting").forEach((row) => wireSetting(row, name));
}
function valueText(s) {
  if (Array.isArray(s.value)) return s.value.join(", ");
  return s.value === null || s.value === undefined ? "" : String(s.value);
}
function settingRow(s) {
  const id = "set_" + s.key.replace(/\W/g, "_");
  const input = s.type === "bool"
    ? `<select id="${id}"><option value="true" ${s.value === true ? "selected" : ""}>true</option><option value="false" ${s.value === false ? "selected" : ""}>false</option></select>`
    : `<input id="${id}" type="${s.type === "list" ? "text" : "number"}" ${s.type === "float" ? 'step="any"' : ""} value="${esc(valueText(s))}">`;
  return `<div class="setting" data-key="${esc(s.key)}" data-type="${s.type}" data-orig="${esc(valueText(s))}">
    <div><div>${esc(s.label)}</div><div class="mono muted">${esc(s.key)}</div></div>${input}
    <span class="src ${s.source}">${s.source}</span>
    <span>${s.source === "profile" ? `<button class="btn sm" data-reset title="Remove from the profile (use the dataset / default value)">Reset</button>` : ""}<button class="btn sm primary" data-setsave hidden>Save</button></span>
    ${s.help ? `<div class="help">${esc(s.help)}</div>` : ""}</div>`;
}
function parseSetting(type, raw) {
  if (type === "bool") return raw === "true";
  if (type === "int") { const v = parseInt(raw, 10); if (!isFinite(v)) throw new Error("enter a whole number"); return v; }
  if (type === "float") { const v = Number(raw); if (!isFinite(v)) throw new Error("enter a number"); return v; }
  const parts = raw.split(",").map((x) => x.trim()).filter(Boolean);
  return parts.every((x) => /^-?\d+$/.test(x)) ? parts.map(Number) : parts;
}
function wireSetting(row, name) {
  const inp = row.querySelector("input, select"), save = row.querySelector("[data-setsave]"), reset = row.querySelector("[data-reset]");
  const changed = () => (save.hidden = inp.value === row.dataset.orig);
  inp.addEventListener("input", changed); inp.addEventListener("change", changed);
  save.onclick = () => {
    let value;
    try { value = parseSetting(row.dataset.type, inp.value); } catch (e) { toast(e.message, true); return; }
    act(() => api("/api/profile/set", { name, key: row.dataset.key, value }), `Saved ${row.dataset.key}`).then(() => loadProfile(name));
  };
  if (reset) reset.onclick = () => act(() => api("/api/profile/set", { name, key: row.dataset.key, unset: true }), `${row.dataset.key}: back to the inherited value`).then(() => loadProfile(name));
}

/* ---------------------------------------------------------------- logs */
async function loadLog(rebuild) {
  const sel = $("logSelect");
  if (rebuild || !sel.options.length) {
    const q = S.queue || (S.queue = await api("/api/queue"));
    const items = [...q.items.filter((i) => i.status === "running"), ...q.items.filter((i) => i.status !== "running" && i.started).slice(-40).reverse()];
    sel.innerHTML = `<option value="runner">Runner</option>` + items.map((i) => `<option value="${esc(i.key)}">${esc(STATUS[i.status] || i.status)} · ${esc(i.label)}</option>`).join("");
    if (![...sel.options].some((o) => o.value === S.logKey)) S.logKey = "runner";
    sel.value = S.logKey;
    sel.onchange = () => { S.logKey = sel.value; loadLog(false); };
  }
  const r = S.logKey === "runner" ? await api("/api/runner_log?lines=300") : await api(`/api/log?key=${encodeURIComponent(S.logKey)}&lines=600`);
  const el = $("logText");
  el.textContent = r.text || "(empty)";
  if ($("logFollow").checked) el.scrollTop = el.scrollHeight;
}

/* ---------------------------------------------------------------- drawer */
function drawer(title, html) {
  $("drawerTitle").textContent = title; $("drawerBody").innerHTML = html; $("drawer").hidden = false;
}
$("drawerClose").onclick = () => ($("drawer").hidden = true);
$("drawer").addEventListener("click", (e) => {
  if (e.target === $("drawer")) $("drawer").hidden = true;
  const q = e.target.closest("[data-dq]");
  if (q) { queueJob(q.dataset.dq, !!q.dataset.rerun); $("drawer").hidden = true; }
});
document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("drawer").hidden = true; });

tick();
