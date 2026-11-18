// 음성 Agent 운영 포털 화면. 상태는 /api/status (서버가 5초 캐시), 동작은 /api/action (허용 목록만, ada-ops/portal_app/server.py).

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };

const STATE = {
  running: ["실행 중", "ok"], demo: ["CPU 시연", "demo"], starting: ["기동 중", "wait"], stopped: ["중지", "off"],
  gpu_wait: ["GPU 대기", "wait"], cpu_ready: ["CPU 가능", "ok2"], policy: ["정책 대기", "policy"],
  decision: ["결정 대기", "decision"], not_built: ["미구현", "off"], disabled: ["비활성", "decision"],
};
const SVC_KO = {
  "ko-gemma-vllm": "Gemma 4 언어 모델 (vLLM :8001)", "ko-realtime-server": "한국어 실시간 음성 서버 (:8766)", "ko-web": "한국어 화면 (:7201)",
  "en-moshi": "Moshi INT8 (:8998)", "en-moshi-bf16": "Moshi BF16 (:8998)", "en-core": "영어 Gateway (:7102)", "en-web": "영어 화면 (:7101)",
  "transcribe-vllm": "Transcribe 30B 전사 (:8002)", "moshi-finetune": "Moshi LoRA 학습",
};
const MODE_KO = {
  realtime: ["실시간 운영", "낮 시간 학생 서비스. 지금은 Moshi INT8 이 꺼져 있어서, 영어 화면과 Gateway 는 뜨지만 영어 대화는 되지 않습니다. 실제로 쓸 수 있는 것은 한국어 Agent 입니다."],
  "en-solo": ["영어 단독", "영어 Agent 만 운영하는 날. Moshi BF16 (약 24 GB)."],
  batch: ["야간 전사", "Transcribe 30B 로 녹음 세션을 축어 전사합니다. nightly-batch 가 올립니다."],
  summary: ["야간 요약", "Gemma 4 로 한국어 세션 요약을 만듭니다. nightly-batch 가 올립니다."],
  train: ["학습", "Moshi 튜터 말투 LoRA 학습."],
};
const SCREEN_INFO = {
  ko_web: ["한국어 Agent", "아동용 한국어 음성 대화 화면"],
  en_web: ["영어 Agent", "Moshi 전이중 영어 대화 화면"],
  studio: ["튜터 녹음 스튜디오", "사람 튜터·학생 대화 녹음 (학습 데이터 수집, GPU 불필요)"],
};
let S = null;

function badge(state) {
  const [t, c] = STATE[state] || [state, "off"];
  return el("span", `badge ${c}`, t);
}

function banner(text, kind = "warn") {
  const b = $("banner");
  b.hidden = !text;
  if (text) { b.className = `banner ${kind}`; b.textContent = text; }
}

function show(view) {
  for (const s of document.querySelectorAll("main > section")) s.hidden = s.id !== `view-${view}`;
  for (const a of document.querySelectorAll("#nav a")) a.classList.toggle("on", a.dataset.view === view);
}

function mib(v) { return v == null ? "-" : `${(v / 1024).toFixed(1)} GB`; }

// ---------- 전체 흐름 ----------
function renderFlow() {
  const lg = $("legend");
  lg.replaceChildren();
  for (const k of ["running", "demo", "gpu_wait", "cpu_ready", "policy", "decision", "stopped", "not_built"]) lg.append(badge(k));
  const f = $("flow");
  f.replaceChildren();
  S.flow.forEach((stage, i) => {
    if (i) f.append(el("div", "arrow", "→"));
    const col = el("div", "stage");
    col.append(el("div", "stage-no", `${i + 1}단계`), el("h2", null, stage.stage), el("p", "small", stage.desc));
    for (const n of stage.nodes) {
      const box = el("div", `node st-${n.state}`);
      const head = el("div", "node-head");
      head.append(el("b", null, n.title), badge(n.state));
      box.append(head, el("p", "small", n.detail));
      const links = el("div", "node-links");
      if (n.link) {
        const a = el("a", null, `열기 :${n.link}`);
        a.href = `http://localhost:${n.link}/`; a.target = "_blank"; a.rel = "noopener";
        links.append(a);
      }
      if (n.menu) {
        const a = el("a", null, "자세히");
        a.href = `#${n.menu}`;
        links.append(a);
      }
      box.append(links);
      col.append(box);
    }
    f.append(col);
  });
  $("updated").textContent = `상태 확인 ${S.time} (10초마다 갱신)`;
}

// ---------- 서비스 운영 ----------
function renderServices() {
  const g = S.gpu;
  const gauge = $("gauge");
  gauge.replaceChildren();
  if (g) {
    const seg = (cls, v, label) => { const d = el("div", `seg ${cls}`); d.style.width = `${(100 * v) / g.total}%`; d.title = `${label} ${mib(v)}`; return d; };
    gauge.append(seg("others", g.others, "다른 사용자"), seg("ours", g.ours, "우리 서비스"), seg("free", g.free, "비어 있음"));
    $("gauge-text").textContent = `전체 ${mib(g.total)} · 다른 사용자 ${mib(g.others)} · 우리 ${mib(g.ours)} · 비어 있음 ${mib(g.free)}`
      + (S.gpu_wait.others_peak != null ? ` · 다른 사용자 최근 2시간 최고 ${mib(S.gpu_wait.others_peak)} → 우리가 쓸 수 있다고 보는 양 ${mib(Math.max(0, S.gpu_avail_for_us))}` : "");
  } else {
    $("gauge-text").textContent = "GPU 상태를 읽지 못했습니다.";
  }
  const gw = S.gpu_wait;
  $("gw-state").replaceChildren(gw.active ? badge("gpu_wait") : badge("stopped"),
    document.createTextNode(gw.active ? ` ${gw.mode} 모드 예약이 감시 중입니다 (pid ${gw.pid}).` : " 예약 없음"));
  $("gw-last").textContent = gw.last ? `마지막 판단: ${gw.last}` : "";
  $("gpu-mini").textContent = g ? `GPU 비어 있음 ${mib(g.free)} / ${mib(g.total)}${S.current_mode ? ` · 모드 ${S.current_mode}` : ""}${gw.active ? ` · ${gw.mode} 예약 중` : ""}` : "GPU -";

  const box = $("modes");
  box.replaceChildren();
  for (const c of S.modes) {
    const [name, desc] = MODE_KO[c.mode] || [c.mode, ""];
    const card = el("div", "card mode");
    const h = el("div", "node-head");
    h.append(el("h3", null, `${name} (${c.mode})`), c.current ? badge("running") : c.reserved ? badge("gpu_wait") : badge("stopped"));
    card.append(h, el("p", "small", desc));
    card.append(el("p", null, `새로 필요한 GPU: ${c.need == null ? "-" : mib(c.need)}${c.fits_now ? " (지금 기준으로 여유 있음)" : " (지금 기준으로 부족)"}`));
    const ul = el("ul", "svcs");
    for (const s of c.services) {
      const li = el("li");
      li.append(badge(s.state), document.createTextNode(` ${SVC_KO[s.name] || s.name}${s.mib ? ` · ${mib(s.mib)}` : ""}`));
      ul.append(li);
    }
    card.append(ul);
    if (c.partial.length) card.append(el("p", "note warn", `이 모드를 올려도 다음 서비스는 올라가지 않습니다: ${c.partial.map((n) => SVC_KO[n] || n).join(", ")}`));
    if (c.reservable) {
      const b = el("button", null, c.reserved ? "예약됨 (감시 중)" : "GPU 여유가 생기면 자동 기동 예약");
      b.disabled = c.reserved;
      b.onclick = () => reserve(c.mode);
      card.append(b);
    } else {
      card.append(el("p", "small", `예약 불가: ${c.reason}`));
    }
    box.append(card);
  }
}

async function reserve(mode) {
  const gw = S.gpu_wait;
  if (gw.active && !confirm(`지금 ${gw.mode} 예약이 감시 중입니다. 이 예약을 멈추고 ${mode} 로 바꿀까요?`)) return;
  if (!gw.active && !confirm(`${mode} 모드를 GPU 여유가 생기면 자동으로 올리도록 예약할까요?`)) return;
  await act({ action: "reserve", mode, replace: gw.active });
}

// ---------- 학생 화면 ----------
function renderScreens() {
  const box = $("screens");
  box.replaceChildren();
  for (const [key, port] of Object.entries(S.screens)) {
    const [name, desc] = SCREEN_INFO[key];
    const up = S.ports[key];
    const card = el("div", "card");
    const h = el("div", "node-head");
    h.append(el("h3", null, name), badge(up ? (S.cpu_demo && key !== "studio" ? "demo" : "running") : "stopped"));
    card.append(h, el("p", "small", desc));
    const a = el("a", "button", `열기 http://localhost:${port}`);
    a.href = `http://localhost:${port}/`; a.target = "_blank"; a.rel = "noopener";
    if (!up) a.classList.add("disabled");
    card.append(a);
    if (key === "studio") {
      const b = el("div", "buttons");
      const s1 = el("button", "ghost", "스튜디오 켜기"); s1.dataset.act = "studio_start";
      const s2 = el("button", "ghost", "스튜디오 끄기"); s2.dataset.act = "studio_stop";
      b.append(s1, s2);
      card.append(b);
    } else if (!up) {
      card.append(el("p", "note", key === "en_web" ? "영어 Agent 는 en-solo 예약 또는 CPU 시연으로 켤 수 있습니다." : "실시간 운영 예약 또는 CPU 시연으로 켤 수 있습니다."));
    }
    box.append(card);
  }
  $("demo-state").replaceChildren(badge(S.cpu_demo ? "demo" : "stopped"), document.createTextNode(S.cpu_demo ? " 시연 실행 중" : " 시연 꺼짐"));
  const ports = [7100, ...Object.values(S.screens)];
  $("tunnel").textContent = `ssh -p 2022 ${ports.map((p) => `-L ${p}:127.0.0.1:${p}`).join(" ")} yang@119.65.103.62`;
}

// ---------- 녹음·데이터 ----------
const STATUS_KO = { queued: "처리 대기", transcribed: "전사 완료", summarized: "요약 완료", created: "녹음 전", recording: "녹음 중", recorded: "저장됨", failed: "실패" };
function renderData() {
  const box = $("queues");
  box.replaceChildren();
  const items = [
    ["korean", "한국어 세션 녹음", S.record.korean, "korean_agent/queue/korean"],
    ["english", "영어 세션 녹음", S.record.english, "english-agent/queue/english"],
    ["tutor", "튜터 녹음 스튜디오", null, "english-agent/queue/tutor"],
  ];
  for (const [k, name, rec, path] of items) {
    const q = S.queues[k];
    const card = el("div", "card");
    const h = el("div", "node-head");
    h.append(el("h3", null, name), rec === null ? badge(S.studio ? "running" : "stopped") : badge(rec ? "running" : "policy"));
    card.append(h);
    card.append(el("p", "big", `${q.total}개 세션`));
    const ul = el("ul", "svcs");
    for (const [st, n] of Object.entries(q.by_status)) ul.append(el("li", null, `${STATUS_KO[st] || st}: ${n}`));
    if (q.child) ul.append(el("li", null, `아동 세션: ${q.child} (학습 활용 ${S.allow_child_data ? "허용" : "보류"})`));
    card.append(ul);
    if (rec === false) card.append(el("p", "note", "녹음 동의·보관 기간 정책이 정해질 때까지 녹음을 꺼 두었습니다 (record: false). 그래서 비어 있는 것이 정상입니다."));
    card.append(el("p", "small mono", path));
    if (k === "tutor") {
      const a = el("a", "button", "스튜디오 열기");
      a.href = `http://localhost:${S.screens.studio}/`; a.target = "_blank"; a.rel = "noopener";
      card.append(a);
    }
    box.append(card);
  }
}

// ---------- 학습 ----------
function renderTrain() {
  const ol = $("train-steps");
  ol.replaceChildren();
  const tq = S.queues.tutor;
  const steps = [
    ["튜터·학생 대화 녹음", S.studio ? "running" : "stopped", `튜터 녹음 스튜디오에서 녹음합니다. 지금 ${tq.total}개 세션이 있습니다.`, null],
    ["튜터 음성 단어 타임스탬프", "decision", "Transcribe 30B 를 GPU 에 올려 단어별 시각을 뽑습니다. 미들웨어 출력 형식을 아직 확인하지 못했습니다.", "(형식 확인 뒤 추가 예정) → english-agent/queue/tutor/<세션>/tutor_words.json"],
    ["학습 데이터 변환", "cpu_ready", `스테레오 wav(왼쪽 튜터, 오른쪽 학생) + 단어 정렬 json + 목록 jsonl 을 만듭니다. 학습 동의가 없거나, 아동인데 활용이 보류되어 있거나, 시계 차이가 크면 뺍니다. 지금 학습 데이터 ${S.finetune.items}개.`,
      "cd ~/pgm/voice_agent/english-agent && .venv-core/bin/python -m finetune.prepare from-studio"],
    ["데이터 검사", "cpu_ready", "길이, 채널, 정렬 파일을 검사합니다.", ".venv-core/bin/python -m finetune.prepare check"],
    ["LoRA 학습", "not_built", "moshi-finetune (H100 기준 약 40 GB). 학습 서비스와 GPU 시간대가 필요합니다. 설정 초안: finetune/moshi_lora.yaml", "torchrun --nproc-per-node 1 -m train finetune/moshi_lora.yaml   (~/opt/moshi-finetune)"],
    ["평가 후 적용", "decision", "학습한 LoRA 로 튜터 말투가 좋아졌는지 평가한 뒤 실시간 Moshi 에 적용할지 정합니다.", null],
  ];
  for (const [t, st, d, cmd] of steps) {
    const li = el("li");
    const h = el("div", "node-head");
    h.append(el("b", null, t), badge(st));
    li.append(h, el("p", "small", d));
    if (cmd) li.append(el("pre", null, cmd));
    ol.append(li);
  }
}

// ---------- 결정 대기 ----------
function renderDecisions() {
  const box = $("decisions");
  box.replaceChildren();
  for (const d of S.decisions) {
    const item = el("div", "decision-item");
    const h = el("div", "node-head");
    h.append(el("b", null, d.title), badge("decision"));
    item.append(h, el("p", "small", d.detail));
    const tags = el("div", "tags");
    for (const b of d.blocks) tags.append(el("span", "tag", b));
    item.append(tags);
    box.append(item);
  }
}

// ---------- 공통 ----------
async function act(body) {
  banner("실행 중입니다…", "info");
  document.body.classList.add("busy");
  try {
    const r = await fetch("/api/action", { method: "POST", headers: { "Content-Type": "application/json", "X-Portal": "1" }, body: JSON.stringify(body) });
    const j = await r.json().catch(() => ({ ok: false, error: `서버 오류 ${r.status}` }));
    banner(j.ok ? "완료했습니다." : j.error || "실패했습니다.", j.ok ? "info" : "warn");
    if (j.output) { $("out-title").textContent = j.ok ? "실행 결과" : "실행 결과 (실패)"; $("out-text").textContent = j.output; $("out-dlg").showModal(); }
  } finally {
    document.body.classList.remove("busy");
    await refresh(true);
  }
}

const CONFIRM = {
  unreserve: "GPU 자동 기동 예약을 취소할까요? 이미 올라간 서비스는 그대로 둡니다.",
  gpu_stop: "우리가 올린 GPU 서비스를 모두 중지할까요? 학생이 쓰는 중이면 대화가 끊깁니다. (다른 사용자의 프로그램은 건드리지 않습니다)",
  demo_start: "CPU 시연을 켤까요? 준비에 최대 3분이 걸립니다.",
  demo_stop: "CPU 시연을 끌까요?",
  studio_start: "튜터 녹음 스튜디오를 켤까요?",
  studio_stop: "튜터 녹음 스튜디오를 끌까요? 녹음 중인 세션은 그 시점까지 저장하고 끝냅니다.",
};
document.addEventListener("click", (e) => {
  const b = e.target.closest("button[data-act]");
  if (!b) return;
  if (confirm(CONFIRM[b.dataset.act] || "실행할까요?")) act({ action: b.dataset.act });
});

async function refresh(fresh = false) {
  try {
    const r = await fetch(`/api/status${fresh ? "?fresh=1" : ""}`);
    S = await r.json();
  } catch (err) {
    banner(`포털 서버에 연결하지 못했습니다: ${err.message}. SSH 터널을 확인하세요.`);
    return;
  }
  renderFlow(); renderServices(); renderScreens(); renderData(); renderTrain(); renderDecisions();
}

function route() { show((location.hash || "#flow").slice(1)); }
window.addEventListener("hashchange", route);
route();
refresh();
setInterval(() => { if (!document.body.classList.contains("busy")) refresh(); }, 10000);
