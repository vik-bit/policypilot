// PolicyPilot UI — plain JavaScript, no build step.
// Pages are <section class="view"> elements switched by the URL hash (#/overview, #/analyze, ...).
// Flow: choose regulation → POST /api/analyze → poll /api/jobs/{id} → show stages → render report.

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch {} },
};

const state = {
  health: null, regulations: [], clauses: {}, policies: {},
  selected: null, source: "samples",
  jobId: null, events: 0, stage: -1, timer: null, startedAt: 0,
  report: null, filter: "all",
};

// What each impact type is called in the UI. Plain words, not jargon.
const IMPACT = {
  conflict: { label: "Needs change", tone: "red" },
  gap: { label: "Missing from policy", tone: "amber" },
  aligned: { label: "Already compliant", tone: "green" },
};

// ------------------------------------------------------------------ icons (inline SVG)
const ICON_PATHS = {
  home: '<path d="M3 10.5 12 3l9 7.5V20a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z"/>',
  play: '<path d="M7 4.5v15l12-7.5z"/>',
  file: '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M9 13h6M9 17h4"/>',
  book: '<path d="M4 4.5A1.5 1.5 0 0 1 5.5 3H20v15H5.5A1.5 1.5 0 0 0 4 19.5zM4 19.5A1.5 1.5 0 0 0 5.5 21H20"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
  help: '<circle cx="12" cy="12" r="9"/><path d="M9.5 9a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .9-1 1.6v.6M12 17h.01"/>',
  check: '<path d="M5 12.5 10 17 19 7"/>',
  alert: '<path d="M12 3.5 2.5 20h19z"/><path d="M12 10v4M12 17h.01"/>',
  spark: '<path d="M12 3v4M12 17v4M3 12h4M17 12h4M6.3 6.3l2.5 2.5M15.2 15.2l2.5 2.5M6.3 17.7l2.5-2.5M15.2 8.8l2.5-2.5"/>',
  shield: '<path d="M12 3 5 6v5c0 4.5 3 8.5 7 10 4-1.5 7-5.5 7-10V6z"/><path d="m9 12 2 2 4-4"/>',
  user: '<circle cx="12" cy="8" r="4"/><path d="M4 21c0-4 3.6-7 8-7s8 3 8 7"/>',
  upload: '<path d="M12 16V4M7 9l5-5 5 5M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/>',
  download: '<path d="M12 4v12M7 11l5 5 5-5M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
  chev: '<path d="m9 6 6 6-6 6"/>',
  refresh: '<path d="M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7"/>',
};
const icon = (name, cls = "") => `<svg class="ico ${cls}" viewBox="0 0 24 24" aria-hidden="true">${ICON_PATHS[name] || ""}</svg>`;
function hydrateIcons(root = document) {
  $$("[data-icon]", root).forEach((el) => { if (!el.querySelector("svg")) el.innerHTML = icon(el.dataset.icon); });
}

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), 3200);
}

// "Circular SFR/2026/014 — Amendments to …" → { code: "Circular SFR/2026/014", name: "Amendments to …" }
function splitTitle(title) {
  const i = title.indexOf("—");
  return i < 0 ? { code: "", name: title } : { code: title.slice(0, i).trim(), name: title.slice(i + 1).trim() };
}

// ------------------------------------------------------------------ routing
function route() {
  const [, view = "overview", sub] = (location.hash || "#/overview").split("/");
  const section = $(`#view-${view}`) ? view : "overview";
  $$(".view").forEach((v) => (v.hidden = v.id !== `view-${section}`));
  $$(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.view === section));
  document.title = `${$(`#view-${section}`).dataset.title} · PolicyPilot`;
  if (section === "help" && sub) {
    const target = $(`#help-${sub}`);
    $$(".help-toc a").forEach((a) => a.classList.toggle("active", a.getAttribute("href") === `#/help/${sub}`));
    if (target) requestAnimationFrame(() => target.scrollIntoView({ behavior: "smooth", block: "start" }));
  } else {
    window.scrollTo({ top: 0 });
  }
  if (section === "analyze" && !state.jobId) showChoose();
  // Shareable link to a saved report: #/report/<run-id>
  if (section === "report" && sub && sub !== state.report?.run_id) {
    api(`/api/jobs/${encodeURIComponent(sub)}`).then((job) => job.report && setReport(job.report, false)).catch(() => {});
  }
}
window.addEventListener("hashchange", route);

// ------------------------------------------------------------------ boot
async function boot() {
  hydrateIcons();
  route();
  try {
    state.health = await api("/api/health");
    const card = $("#mode-card");
    const provider = state.health.mode === "groq" ? "Groq" : "Claude";
    if (state.health.key_missing) {
      card.className = "mode-card demo";
      $("#mode-title").textContent = `${provider} selected`;
      $("#mode-text").textContent = `Its API key is missing. Add ${state.health.mode === "groq" ? "GROQ_API_KEY" : "ANTHROPIC_API_KEY"} to .env and restart.`;
    } else if (state.health.mode !== "offline") {
      card.className = "mode-card live";
      $("#mode-title").textContent = "AI mode";
      $("#mode-text").textContent = `${provider} · ${state.health.model}`;
    } else {
      card.className = "mode-card demo";
      $("#mode-title").textContent = "Demo mode";
      $("#mode-text").textContent = "No AI key set up. Search and fact-checking are real; the assessment is rule-based.";
    }
    $("#ov-policies").textContent = state.health.policies.length;
    $("#ov-clauses").textContent = state.health.clauses;
  } catch {
    $("#mode-title").textContent = "Server offline";
    $("#mode-text").textContent = "Start it with: python -m policypilot serve";
  }

  state.regulations = await api("/api/regulations");
  $("#ov-regs").textContent = state.regulations.length;
  renderRegList();
  await loadPolicies();

  // Re-open the last report after a page refresh.
  const last = store.get("pp-last-run");
  if (last) {
    try {
      const job = await api(`/api/jobs/${last}`);
      if (job.report) setReport(job.report, false);
    } catch {}
  }
}

// ------------------------------------------------------------------ step 1: choose
function renderRegList() {
  $("#reg-list").innerHTML = state.regulations.map((r) => {
    const t = splitTitle(r.title);
    const eff = r.meta?.effective ? ` · Effective ${esc(r.meta.effective)}` : "";
    return `<button class="reg-row" role="option" data-id="${esc(r.id)}">
        <span class="radio" aria-hidden="true"></span>
        <span><span class="reg-name">${esc(t.name)}</span><span class="reg-sub">${esc(t.code)}${eff}</span></span>
        <span class="pill pill-grey">${r.requirements} requirements</span>
      </button>`;
  }).join("");
  $$(".reg-row").forEach((row) => row.addEventListener("click", () => selectReg(row.dataset.id)));
}

function selectReg(id) {
  state.selected = id;
  $$(".reg-row").forEach((r) => {
    r.classList.toggle("selected", r.dataset.id === id);
    r.setAttribute("aria-selected", r.dataset.id === id);
  });
  const r = state.regulations.find((x) => x.id === id);
  const t = splitTitle(r.title);
  const m = r.meta || {};
  $("#reg-preview").innerHTML = `
    <span class="pill pill-blue">${esc(t.code || "Regulation")}</span>
    <h3 style="margin-top:10px">${esc(t.name)}</h3>
    <dl class="meta-list">
      ${m.issued_by ? `<dt>Issued by</dt><dd>${esc(m.issued_by)}</dd>` : ""}
      ${m.effective ? `<dt>Effective</dt><dd>${esc(m.effective)}</dd>` : ""}
      ${m.applies_to ? `<dt>Applies to</dt><dd>${esc(m.applies_to)}</dd>` : ""}
    </dl>
    <div class="f-label">What it requires</div>
    <ul class="req-list">${r.requirement_list.map((q) => `<li><span class="rid">${esc(q.id)}</span><span>${esc(q.title)}</span></li>`).join("")}</ul>`;
  updateRunButton();
}

$$(".segmented button").forEach((b) => b.addEventListener("click", () => {
  state.source = b.dataset.source;
  $$(".segmented button").forEach((x) => x.classList.toggle("on", x === b));
  $("#source-samples").hidden = state.source !== "samples";
  $("#source-paste").hidden = state.source !== "paste";
  $("#reg-preview").hidden = state.source !== "samples";
  $(".choose").style.gridTemplateColumns = state.source === "paste" ? "minmax(0,1fr)" : "";
  updateRunButton();
}));
$("#reg-text").addEventListener("input", updateRunButton);

function updateRunButton() {
  const pasted = $("#reg-text").value.trim();
  const ready = state.source === "samples" ? !!state.selected : pasted.length > 20;
  $("#run-btn").disabled = !ready;
  if (state.source === "samples") {
    const r = state.regulations.find((x) => x.id === state.selected);
    $("#action-hint").textContent = r
      ? `Ready to check ${r.requirements} requirements against ${state.health?.clauses ?? "all"} policy clauses.`
      : "Select a regulation to continue.";
  } else {
    $("#action-hint").textContent = ready ? "Ready to analyse your text." : "Paste at least a sentence or two.";
  }
}

function setStepper(step) {
  $$("#stepper li").forEach((li) => {
    const n = Number(li.dataset.step);
    li.classList.toggle("done", n < step);
    li.classList.toggle("current", n === step);
    $(".step-dot", li).innerHTML = n < step ? icon("check") : n;
  });
}

function showChoose() {
  setStepper(1);
  $("#step-choose").hidden = false;
  $("#action-bar").hidden = false;
  $("#step-run").hidden = true;
}

// ------------------------------------------------------------------ step 2: run
$("#run-btn").addEventListener("click", async () => {
  const body = state.source === "samples" ? { regulation_id: state.selected } : { text: $("#reg-text").value };
  const reg = state.regulations.find((x) => x.id === state.selected);
  $("#run-title").textContent = state.source === "samples" ? `Analysing “${splitTitle(reg.title).name}”` : "Analysing your regulation";
  $("#run-sub").textContent = (state.health && state.health.mode !== "offline")
    ? "The AI reads each clause before deciding. This usually takes 1–2 minutes."
    : "Demo mode is quick — usually under 30 seconds.";
  setStepper(2);
  $("#step-choose").hidden = true;
  $("#action-bar").hidden = true;
  $("#step-run").hidden = false;
  $("#trace").innerHTML = "";
  $("#log-count").textContent = "";
  state.events = 0;
  state.stage = -1;
  state.searches = 0;
  $$("#stages li").forEach((li) => { li.className = ""; $(".stage-dot", li).innerHTML = ""; });
  $("#stage-search-detail").textContent = "Finding clauses that relate to each requirement";
  $("#stage-verify-detail").textContent = "Checking the evidence word-for-word";
  advanceStage(0);
  startTimer();
  try {
    const { job_id } = await api("/api/analyze", { method: "POST", body: JSON.stringify(body) });
    state.jobId = job_id;
    poll();
  } catch (err) {
    failRun(err.message);
  }
});

function startTimer() {
  clearInterval(state.timer);
  state.startedAt = Date.now();
  const tick = () => {
    const s = Math.floor((Date.now() - state.startedAt) / 1000);
    $("#run-timer").textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  };
  tick();
  state.timer = setInterval(tick, 1000);
}

const STAGES = ["read", "search", "draft", "verify", "done"];
function advanceStage(i) {
  if (i <= state.stage) return;
  state.stage = i;
  $$("#stages li").forEach((li, n) => {
    li.classList.toggle("done", n < i || (i === 4 && n === 4));
    li.classList.toggle("active", n === i && i !== 4);
    $(".stage-dot", li).innerHTML = li.classList.contains("done") ? icon("check") : "";
  });
}

function handleEvent(e) {
  if (e.kind === "plan") advanceStage(1);
  if (e.kind === "search") {
    advanceStage(1);
    state.searches++;
    $("#stage-search-detail").textContent = `${state.searches} search${state.searches > 1 ? "es" : ""} run so far`;
  }
  if (e.kind === "llm" && state.searches > 0) advanceStage(2);
  if (e.kind === "verify") {
    advanceStage(3);
    if (e.data?.total != null) $("#stage-verify-detail").textContent = `${e.data.valid} of ${e.data.total} quotes confirmed`;
    else $("#stage-verify-detail").textContent = e.message.replace(/^Fact-check:?\s*/i, "");
  }
  if (e.kind === "done") advanceStage(4);
}

async function poll() {
  let job;
  try {
    job = await api(`/api/jobs/${state.jobId}?since=${state.events}`);
  } catch (err) {
    return failRun(err.message);
  }
  for (const e of job.events) {
    const secs = Math.max(0, Math.round((new Date(e.at) - state.startedAt) / 1000));
    $("#trace").insertAdjacentHTML("beforeend", `<li class="k-${esc(e.kind)}"><span class="t">+${secs}s</span><span>${esc(e.message)}</span></li>`);
    state.events++;
    handleEvent(e);
  }
  $("#log-count").textContent = `(${state.events} steps)`;
  if (job.status === "running") return setTimeout(poll, 800);
  clearInterval(state.timer);
  if (job.status === "error") return failRun(job.error);
  advanceStage(4);
  setReport(job.report, true);
  state.jobId = null;
  setTimeout(() => { location.hash = "#/report"; }, 700);
}

function failRun(msg) {
  clearInterval(state.timer);
  const active = $("#stages li.active");
  if (active) { active.classList.remove("active"); active.classList.add("failed"); }
  $("#run-title").textContent = "Something went wrong";
  $("#run-sub").innerHTML = `${esc(msg)} <button class="btn btn-sm btn-ghost" id="retry-btn">Back</button>`;
  $("#retry-btn").addEventListener("click", () => { state.jobId = null; showChoose(); });
  state.jobId = null;
}

// ------------------------------------------------------------------ step 3: report
function setReport(report, fresh) {
  state.report = report;
  state.filter = "all";
  store.set("pp-last-run", report.run_id);
  renderReport();
  if (fresh) toast("Your report is ready");
}

function counts(r) {
  const f = r.findings;
  const by = (k) => f.filter((x) => x.impact === k).length;
  const approved = Object.values(r.decisions).filter((v) => v === "approved").length;
  const rejected = Object.values(r.decisions).filter((v) => v === "rejected").length;
  return {
    total: f.length, conflict: by("conflict"), gap: by("gap"), aligned: by("aligned"),
    unverified: f.filter((x) => !x.verified).length, approved, rejected,
  };
}

function renderReport() {
  const r = state.report;
  const c = counts(r);
  const s = r.stats;
  const t = splitTitle(r.regulation_title);
  const pending = c.total - c.approved - c.rejected;
  const badge = $("#nav-report-badge");
  badge.hidden = !pending;
  badge.textContent = pending;
  const corrected = s.submissions > 1 ? s.final_valid - s.first_pass_valid : 0;
  const when = new Date(r.created_at).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
  const modePill = r.mode !== "offline" ? `<span class="pill pill-green">AI · ${esc(r.model)}</span>` : `<span class="pill pill-amber">Demo mode</span>`;

  const matches = (x) => state.filter === "all" || (state.filter === "unverified" ? !x.verified : x.impact === state.filter);
  const filters = [["all", "All", c.total], ["conflict", "Needs change", c.conflict], ["gap", "Missing", c.gap],
    ["aligned", "Compliant", c.aligned], ["unverified", "Check manually", c.unverified]];

  $("#report").innerHTML = `
    <div class="report-head">
      <div>
        <span class="muted small">${esc(t.code)}</span>
        <h1>${esc(t.name)}</h1>
        <div class="report-meta">${modePill}<span>Analysed ${esc(when)}</span><span>·</span><span>${r.requirements.length} requirements checked</span></div>
      </div>
      <div class="report-actions">
        <button class="btn btn-ghost" id="export-btn">${icon("download")}Export</button>
        <a class="btn btn-primary" href="#/analyze">${icon("refresh")}New analysis</a>
      </div>
    </div>

    ${r.mode === "offline" ? `<div class="demo-note">${icon("info")}<p><b>This report was made in demo mode.</b> The matched clauses and quotes are real and verified, but the status labels are
      a rule-based guess and the suggested wording is a placeholder. <a href="#/help/modes">Turn on AI mode</a> for full explanations and drafted text.</p></div>` : ""}

    <div class="summary-card"><h3>Summary</h3><p>${esc(r.summary || "No summary was produced.")}</p></div>

    <div class="kpis">
      <div class="kpi red"><div class="v">${c.conflict}</div><div class="l">Need change</div></div>
      <div class="kpi amber"><div class="v">${c.gap}</div><div class="l">Missing from policy</div></div>
      <div class="kpi green"><div class="v">${c.aligned}</div><div class="l">Already compliant</div></div>
      <div class="kpi blue"><div class="v">${s.final_valid}/${s.final_citations}</div><div class="l">Quotes verified</div>
        ${corrected > 0 ? `<small>${corrected} corrected by the fact-checker</small>` : ""}</div>
    </div>

    <div class="review-bar">
      <b>Review progress</b>
      <div class="progress" role="progressbar" aria-valuenow="${c.approved + c.rejected}" aria-valuemax="${c.total}">
        <span class="p-ok" style="width:${c.total ? (100 * c.approved) / c.total : 0}%"></span>
        <span class="p-no" style="width:${c.total ? (100 * c.rejected) / c.total : 0}%"></span>
      </div>
      <span class="muted">${c.approved} approved · ${c.rejected} rejected · ${pending} to review</span>
    </div>

    <div class="filter-row">
      <div class="chips" style="margin:0">${filters.map(([k, label, n]) =>
        `<button class="chip ${state.filter === k ? "on" : ""}" data-f="${k}">${label}<span class="n">${n}</span></button>`).join("")}</div>
    </div>

    ${r.requirements.map((req) => {
      const items = r.findings.filter((x) => x.requirement_id === req.id);
      const shown = items.filter(matches);
      if (state.filter !== "all" && !shown.length) return "";
      return `<section class="req-block">
        <div class="req-block-head"><span class="req-tag">${esc(req.id)}</span><h3>${esc(req.title)}</h3>
          <span class="count">${items.length} clause${items.length === 1 ? "" : "s"} affected</span></div>
        <div class="req-text">${esc(req.text)}</div>
        ${shown.length ? shown.map(findingCard).join("") : `<p class="none">No policy clauses were found to be affected by this requirement.</p>`}
      </section>`;
    }).join("")}`;

  $$("#report .chip").forEach((b) => b.addEventListener("click", () => { state.filter = b.dataset.f; renderReport(); }));
  $$("#report [data-decide]").forEach((b) => b.addEventListener("click", () => decide(b.dataset.key, b.dataset.decide)));
  $("#export-btn").addEventListener("click", exportReport);
  $("#report").hidden = false;
  $("#report-empty").hidden = true;
}

function findingCard(x) {
  const c = state.clauses[x.clause_id];
  const key = `${x.requirement_id}|${x.clause_id}`;
  const verdict = state.report.decisions[key];
  const imp = IMPACT[x.impact];
  const current = c ? highlight(c.text, x.policy_quote) : `<i>This clause could not be found in the library.</i>`;
  const proposed = x.impact === "aligned" || !x.proposed_text
    ? `<i class="muted">No change needed.</i>`
    : c ? wordDiff(c.text, x.proposed_text) : esc(x.proposed_text);
  return `
    <article class="finding ${imp.tone} ${verdict === "rejected" ? "is-rejected" : ""}">
      <div class="f-top">
        <span class="pill pill-${imp.tone}">${imp.label}</span>
        ${x.verified
          ? `<span class="f-evidence ok" title="Every quote below was found word-for-word in the original documents">${icon("check")}Evidence verified</span>`
          : `<span class="f-evidence bad" title="A quote could not be confirmed. Read the original before approving.">${icon("alert")}Check manually</span>`}
      </div>
      <div class="f-title">${c ? `${esc(c.policy_title.replace(/^.*?—\s*/, ""))} · ${esc(c.number)} ${esc(c.title)}` : esc(x.clause_id)}
        <small>Clause ${esc(x.clause_id)}</small></div>
      <div class="f-label">Why it matters</div>
      <p class="f-explain">${esc(x.explanation)}</p>
      <div class="f-label">New rule says</div>
      <div class="f-quote">“${esc(x.regulation_quote)}”</div>
      <div class="f-compare" style="margin-top:14px">
        <div><div class="f-label">Policy today</div>${current}</div>
        <div><div class="f-label">Suggested wording</div>${proposed}</div>
      </div>
      ${x.issues.length ? `<div class="f-issues"><b>Why this needs a manual check</b><ul>${x.issues.map((i) => `<li>${esc(i)}</li>`).join("")}</ul></div>` : ""}
      <div class="f-actions">
        <button class="btn btn-sm btn-ghost btn-approve ${verdict === "approved" ? "on" : ""}" data-key="${esc(key)}" data-decide="${verdict === "approved" ? "pending" : "approved"}">${icon("check")}Approve</button>
        <button class="btn btn-sm btn-ghost btn-reject ${verdict === "rejected" ? "on" : ""}" data-key="${esc(key)}" data-decide="${verdict === "rejected" ? "pending" : "rejected"}">Reject</button>
        <span class="state">${verdict ? `You ${verdict} this change` : "Awaiting your review"}</span>
      </div>
    </article>`;
}

async function decide(key, verdict) {
  try {
    const res = await api("/api/decisions", { method: "POST", body: JSON.stringify({ run_id: state.report.run_id, key, verdict }) });
    state.report.decisions = res.decisions;
    const y = window.scrollY;
    renderReport();
    window.scrollTo({ top: y });
  } catch (err) {
    toast(`Couldn't save: ${err.message}`);
  }
}

// Highlight a (possibly "..."-joined) quote inside text, ignoring case and spacing differences.
function highlight(text, quote) {
  let html = esc(text);
  for (const frag of (quote || "").split(/\.\.\.|…/).map((s) => s.trim().replace(/^["“]|["”.,;:]$/g, "")).filter((s) => s.length > 3)) {
    const pattern = esc(frag).replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\s+/g, "\\s+");
    html = html.replace(new RegExp(pattern, "i"), (m) => `<mark>${m}</mark>`);
  }
  return html;
}

// Word-level diff (longest common subsequence) so reviewers see exactly what changes.
function wordDiff(a, b) {
  const A = a.split(/\s+/), B = b.split(/\s+/);
  if (A.length * B.length > 250000) return esc(b);
  const dp = Array.from({ length: A.length + 1 }, () => new Uint16Array(B.length + 1));
  for (let i = A.length - 1; i >= 0; i--)
    for (let j = B.length - 1; j >= 0; j--)
      dp[i][j] = A[i] === B[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  if (dp[0][0] < Math.min(A.length, B.length) * 0.25) return esc(b); // too different: a diff would be noise
  const out = [];
  let i = 0, j = 0;
  while (i < A.length && j < B.length) {
    if (A[i] === B[j]) { out.push(esc(A[i])); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) out.push(`<del>${esc(A[i++])}</del>`);
    else out.push(`<ins>${esc(B[j++])}</ins>`);
  }
  while (i < A.length) out.push(`<del>${esc(A[i++])}</del>`);
  while (j < B.length) out.push(`<ins>${esc(B[j++])}</ins>`);
  return out.join(" ");
}

// Download the report as a Markdown document that opens in any editor.
function exportReport() {
  const r = state.report;
  const c = counts(r);
  const lines = [
    `# Regulatory impact report — ${r.regulation_title}`, "",
    `- Analysed: ${new Date(r.created_at).toLocaleString()}`,
    `- Mode: ${r.mode !== "offline" ? `AI (${r.model})` : "Demo (rule-based)"}`,
    `- Results: ${c.conflict} need change · ${c.gap} missing from policy · ${c.aligned} already compliant`,
    `- Quotes verified: ${r.stats.final_valid}/${r.stats.final_citations}`,
    `- Review: ${c.approved} approved · ${c.rejected} rejected · ${c.total - c.approved - c.rejected} pending`, "",
    "## Summary", "", r.summary || "—", "",
  ];
  for (const req of r.requirements) {
    lines.push(`## ${req.id}. ${req.title}`, "", `> ${req.text}`, "");
    const items = r.findings.filter((x) => x.requirement_id === req.id);
    if (!items.length) lines.push("_No affected clauses._", "");
    for (const x of items) {
      const cl = state.clauses[x.clause_id];
      const verdict = r.decisions[`${x.requirement_id}|${x.clause_id}`] || "pending";
      lines.push(
        `### ${cl ? `${cl.number} ${cl.title}` : x.clause_id} — ${IMPACT[x.impact].label}`, "",
        `- Clause: \`${x.clause_id}\` · Evidence: ${x.verified ? "verified" : "CHECK MANUALLY"} · Decision: **${verdict}**`, "",
        `**Why it matters:** ${x.explanation}`, "",
        `**New rule says:** "${x.regulation_quote}"`, "",
        `**Policy today:** "${x.policy_quote || "—"}"`, "",
        `**Suggested wording:** ${x.proposed_text || "No change needed."}`, "",
      );
    }
  }
  lines.push("---", "_Generated by PolicyPilot. Drafting aid only; not legal advice._");
  const blob = new Blob([lines.join("\n")], { type: "text/markdown" });
  const a = Object.assign(document.createElement("a"), {
    href: URL.createObjectURL(blob), download: `policypilot-report-${r.regulation_id}.md`,
  });
  a.click();
  URL.revokeObjectURL(a.href);
  toast("Report downloaded");
}

// ------------------------------------------------------------------ policy library
async function loadPolicies() {
  const p = await api("/api/policies");
  state.policies = p.clauses;
  state.clauses = {};
  for (const list of Object.values(p.clauses)) for (const c of list) state.clauses[c.id] = c;
  renderLibrary();
  $("#library-meta").textContent = `${p.stats.clauses} clauses across ${p.stats.policies.length} policies`;
}

function renderLibrary() {
  const q = $("#library-filter").value.trim().toLowerCase();
  const html = Object.values(state.policies).map((list) => {
    const shown = q ? list.filter((c) => `${c.number} ${c.title} ${c.text}`.toLowerCase().includes(q)) : list;
    if (!shown.length) return "";
    const title = list[0].policy_title.replace(/^.*?—\s*/, "");
    return `<details class="policy" ${q ? "open" : ""}>
      <summary><span class="policy-icon">${icon("book")}</span>
        <span><b>${esc(title)}</b><span class="muted">${list.length} clauses${q ? ` · ${shown.length} match` : ""}</span></span>
        ${icon("chev", "chev")}</summary>
      ${shown.map((c) => `<div class="clause"><div class="clause-num">${esc(c.number)}<code>${esc(c.id)}</code></div>
        <div class="clause-body"><b>${esc(c.title)}</b><p>${esc(c.text)}</p></div></div>`).join("")}
    </details>`;
  }).join("");
  $("#policies").innerHTML = html || `<p class="muted">No clauses match “${esc(q)}”.</p>`;
}
$("#library-filter").addEventListener("input", renderLibrary);

$("#upload").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  toast(`Uploading ${file.name}…`);
  const res = await fetch("/api/policies", { method: "POST", body: form });
  const data = await res.json();
  e.target.value = "";
  if (!res.ok) return toast(`Upload failed: ${data.detail}`);
  toast(`Added ${data.clauses_added} clauses from ${data.saved}`);
  await loadPolicies();
});

// ------------------------------------------------------------------ search playground
const METHOD_NOTES = {
  best: "Combines exact-word and meaning search, then a second model re-reads the top results to put the most relevant first. This is what the analysis uses.",
  hybrid: "Merges the exact-word and meaning rankings: a clause near the top of either list rises to the top.",
  bm25: "Ranks clauses by shared words, giving more weight to rare words. Great for exact terms like “V-CIP” or “AES-256”; misses paraphrases.",
  dense: "Ranks clauses by meaning, so “how long we keep data” finds “Data Retention” even without shared words. Can blur exact terms.",
};
$("#search-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("#search-q").value.trim();
  const m = $("#search-method").value;
  $("#search-results").innerHTML = `<p class="muted">Searching…</p>`;
  try {
    const hits = await api(`/api/search?q=${encodeURIComponent(q)}&k=6&method=${m}`);
    $("#search-results").innerHTML = `<div class="explain-box"><b>How this method works:</b> ${METHOD_NOTES[m]}</div>` +
      (hits.length ? hits.map((h, n) => `
        <div class="result ${n === 0 ? "top" : ""}">
          <span class="result-rank">${n + 1}</span>
          <div>
            <div class="result-head"><b>${esc(h.clause.number)} ${esc(h.clause.title)}</b>
              <span class="muted small">${esc(h.clause.policy_title.replace(/^.*?—\s*/, ""))}</span></div>
            <p>${esc(h.clause.text)}</p>
            <div class="rank-chips">
              <span class="pill pill-grey">Exact-words rank: ${h.ranks.bm25 ? "#" + h.ranks.bm25 : "not in top 30"}</span>
              <span class="pill pill-grey">Meaning rank: ${h.ranks.dense ? "#" + h.ranks.dense : "not in top 30"}</span>
            </div>
          </div>
        </div>`).join("") : `<p class="muted">No matching clauses.</p>`);
  } catch (err) {
    $("#search-results").innerHTML = `<p class="muted">Search failed: ${esc(err.message)}</p>`;
  }
});
$$("#search-examples .chip").forEach((b) => b.addEventListener("click", () => {
  $("#search-q").value = b.textContent;
  $("#search-form").requestSubmit();
}));

boot();
