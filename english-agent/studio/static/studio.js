// 튜터 녹음 스튜디오. 역할(tutor, student)마다 WebSocket 하나로 PCM16 24 kHz 모노를 보낸다 (studio/server.py).
// 기기 1대 모드(role=both)는 같은 규약으로 마이크 두 개, 소켓 두 개를 연다.

const RATE = 24000;
const $ = (id) => document.getElementById(id);
const qs = new URLSearchParams(location.search);
const ROLE_KO = { tutor: "튜터", student: "학생", turns: "공용" };
// 순차 녹음(role=turns)은 마이크 하나, 소켓 하나로 보내고 "튜터 완료"·"학생 완료"로 차례를 바꾼다.

function banner(text, kind = "warn") {
  const b = $("banner");
  if (!text) { b.hidden = true; return; }
  b.className = `banner ${kind}`;
  b.textContent = text;
  b.hidden = false;
}

function show(view) {
  for (const v of ["new", "list", "guide", "room"]) $(`view-${v}`).hidden = v !== view;
  for (const a of document.querySelectorAll("#nav a")) a.classList.toggle("on", a.dataset.view === view);
}

function fmtDur(s) {
  if (s == null) return "-";
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
}

// ---------- 새 세션 ----------
function setupForm() {
  const f = $("form");
  const sync = () => {
    const child = f.student_group.value === "child";
    $("age-wrap").hidden = !child;
    $("guardian-wrap").hidden = !child;
  };
  for (const r of f.student_group) r.addEventListener("change", sync);
  sync();
  f.addEventListener("submit", async (e) => {
    e.preventDefault();
    const body = {};
    for (const el of f.elements) {
      if (!el.name) continue;
      if (el.type === "checkbox") body[el.name] = el.checked;
      else if (el.type === "radio") { if (el.checked) body[el.name] = el.value; }
      else body[el.name] = el.value;
    }
    $("form-errors").replaceChildren();
    const r = await fetch("/api/sessions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const j = await r.json().catch(() => ({ errors: [`서버 오류 ${r.status}`] }));
    if (!r.ok || !j.ok) {
      for (const msg of j.errors || []) { const li = document.createElement("li"); li.textContent = msg; $("form-errors").append(li); }
      return;
    }
    f.hidden = true;
    $("created").hidden = false;
    $("created-id").textContent = `세션 ${j.id}`;
    const links = body.mode === "turns"
      ? [["순차 녹음 (마이크 1개, 튜터 → 학생 차례대로)", j.links.turns]]
      : body.mode === "one_device"
        ? [["기기 1대 (튜터·학생 마이크 2개)", j.links.both]]
        : [["튜터 화면 (녹음 시작·종료)", j.links.tutor], ["학생 화면 (학생 기기에서 열기)", j.links.student]];
    for (const [label, href] of links) {
      const li = document.createElement("li");
      const a = document.createElement("a");
      a.href = href; a.textContent = label;
      const code = document.createElement("code");
      code.textContent = location.origin + href;
      li.append(a, document.createElement("br"), code);
      $("links").append(li);
    }
  });
}

// ---------- 목록 ----------
async function loadList() {
  const r = await fetch("/api/sessions");
  const j = await r.json();
  $("child-gate").textContent = j.allow_child_data
    ? "아동 녹음 학습 활용: 허용 (내부관리규정 검토 완료로 설정됨)"
    : "아동 녹음 학습 활용: 보류 중. 아동 녹음은 저장되지만 내부관리규정 검토가 끝날 때까지 학습 데이터로 바뀌지 않습니다.";
  const tb = $("rows");
  tb.replaceChildren();
  if (!j.sessions.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 7; td.textContent = "녹음이 없습니다.";
    tr.append(td); tb.append(tr);
  }
  for (const s of j.sessions) {
    const tr = document.createElement("tr");
    const cell = (t) => { const td = document.createElement("td"); td.textContent = t; tr.append(td); return td; };
    const a = document.createElement("a");
    const role = { one_device: "both", turns: "turns" }[s.mode] || "tutor";
    a.href = `/?sid=${s.id}&role=${role}`;
    a.textContent = s.id;
    cell("").append(a);
    cell(`${s.tutor_alias} / ${s.student_alias}`);
    cell(s.student_group === "child" ? `아동 ${s.age_band || ""}` : "성인");
    cell(fmtDur(s.duration_s));
    cell(s.training_consent ? "예" : "아니오");
    cell({ created: "녹음 전", recording: "녹음 중", recorded: "저장됨" }[s.status] || s.status);
    const del = document.createElement("button");
    del.textContent = "삭제";
    del.className = "danger";
    del.disabled = s.status === "recording";
    del.onclick = async () => {
      if (!confirm(`${s.id} 녹음과 그 녹음으로 만든 학습 데이터를 지웁니다. 되돌릴 수 없습니다. 계속할까요?`)) return;
      const d = await fetch(`/api/sessions/${s.id}`, { method: "DELETE" });
      banner(d.ok ? `${s.id} 를 지웠습니다.` : `삭제 실패: ${await d.text()}`, d.ok ? "info" : "warn");
      loadList();
    };
    cell("").append(del);
    tb.append(tr);
  }
}

// ---------- 녹음 방 ----------
const R = { sid: null, role: null, roles: [], meta: null, ctx: null, mics: {}, socks: {}, playHead: 0, timer: null, state: null };

function wsUrl(role, listen) {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/ws/${R.sid}/${role}?listen=${listen ? 1 : 0}`;
}

function enqueue(buf) {
  if (!R.ctx || !$("listen").checked) return;
  const pcm = new Int16Array(buf);
  if (!pcm.length) return;
  const f = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) f[i] = pcm[i] / 0x8000;
  const ab = R.ctx.createBuffer(1, f.length, RATE);
  ab.copyToChannel(f, 0);
  const src = R.ctx.createBufferSource();
  src.buffer = ab;
  src.connect(R.ctx.destination);
  // 0.3초 넘게 밀리면 따라잡는다 (지연이 쌓이지 않게)
  if (R.playHead - R.ctx.currentTime > 0.3) R.playHead = R.ctx.currentTime;
  const at = Math.max(R.ctx.currentTime + 0.04, R.playHead);
  src.start(at);
  R.playHead = at + ab.duration;
}

function micBlock(role) {
  const wrap = document.createElement("div");
  wrap.className = "mic";
  const label = document.createElement("label");
  label.textContent = role === "turns" ? "마이크 (튜터·학생 공용) " : `${ROLE_KO[role]} 마이크 `;
  const sel = document.createElement("select");
  sel.id = `dev-${role}`;
  const meter = document.createElement("meter");
  meter.id = `lvl-${role}`;
  meter.min = 0; meter.max = 1; meter.low = 0.02; meter.high = 0.6; meter.value = 0;
  label.append(sel);
  wrap.append(label, meter);
  return wrap;
}

async function listDevices() {
  const devs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "audioinput");
  R.roles.forEach((role, idx) => {
    const sel = $(`dev-${role}`);
    const prev = sel.value;
    sel.replaceChildren();
    for (const d of devs) {
      const o = document.createElement("option");
      o.value = d.deviceId;
      o.textContent = d.label || `마이크 ${sel.length + 1}`;
      sel.append(o);
    }
    if (prev) sel.value = prev;
    else if (R.roles.length > 1 && devs[idx]) sel.value = devs[idx].deviceId;  // 기기 1대: 서로 다른 마이크를 기본 선택
  });
}

async function connect() {
  $("connect").disabled = true;
  try {
    if (R.roles.length > 1) {
      const a = $("dev-tutor").value, b = $("dev-student").value;
      if (a && a === b) throw new Error("튜터와 학생 마이크를 서로 다른 장치로 고르세요.");
    }
    R.ctx = R.ctx || new AudioContext({ latencyHint: "interactive" });
    await R.ctx.resume();
    await R.ctx.audioWorklet.addModule("/static/capture-worklet.js");
    const listen = R.roles.length === 1 && $("listen").checked;
    const sink = R.ctx.createGain();
    sink.gain.value = 0;
    sink.connect(R.ctx.destination);
    for (const role of R.roles) {
      const devId = $(`dev-${role}`).value;
      const stream = await navigator.mediaDevices.getUserMedia({ audio: {
        deviceId: devId ? { exact: devId } : undefined, channelCount: 1,
        // 학습 데이터 품질을 위해 잡음 제거·자동 음량은 끈다. 상대 음성을 스피커로 들을 때만 에코 제거를 켠다
        echoCancellation: listen, noiseSuppression: false, autoGainControl: false } });
      const src = R.ctx.createMediaStreamSource(stream);
      const node = new AudioWorkletNode(R.ctx, "pcm16-capture", { processorOptions: { targetRate: RATE, chunkMs: 80 } });
      const ws = new WebSocket(wsUrl(role, listen));
      ws.binaryType = "arraybuffer";
      node.port.onmessage = (e) => {
        if (ws.readyState === WebSocket.OPEN) ws.send(e.data);
        const pcm = new Int16Array(e.data);
        let acc = 0;
        for (let i = 0; i < pcm.length; i++) acc += (pcm[i] / 0x8000) ** 2;
        $(`lvl-${role}`).value = Math.min(1, Math.sqrt(acc / pcm.length) * 4);
      };
      src.connect(node);
      node.connect(sink);
      ws.onmessage = (e) => { if (typeof e.data === "string") onEvent(JSON.parse(e.data)); else enqueue(e.data); };
      ws.onclose = () => { if (R.socks[role] === ws) { banner("서버 연결이 끊겼습니다. 새로고침 후 다시 연결하세요."); stopTimer(); } };
      R.mics[role] = { stream, node, src };
      R.socks[role] = ws;
    }
    await listDevices();  // 권한을 받은 뒤에야 장치 이름이 보인다
    $("connect").textContent = "연결됨";
    if (ctrlSock()) { $("rec-start").hidden = false; $("rec-stop").hidden = false; }
    $("listen-wrap").hidden = R.roles.length > 1 || R.role === "turns";
  } catch (err) {
    banner(`마이크 연결 실패: ${err.message || err}. localhost(SSH 터널) 또는 HTTPS 로 접속했는지 확인하세요.`);
    $("connect").disabled = false;
    closeMics();
  }
}

function closeMics() {
  for (const m of Object.values(R.mics)) { m.stream.getTracks().forEach((t) => t.stop()); m.node.disconnect(); }
  for (const ws of Object.values(R.socks)) { try { ws.close(); } catch (_) { /* closed */ } }
  R.mics = {};
  R.socks = {};
}

function ctrlSock() { return R.socks.tutor || R.socks.turns; }  // 녹음 시작·종료·차례를 보낼 소켓

function send(msg) {
  const ws = ctrlSock();
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
}

function stopTimer() { clearInterval(R.timer); R.timer = null; }

function onEvent(ev) {
  if (ev.type === "state") {
    R.state = ev;
    for (const role of ["tutor", "student"]) {
      const on = ev.connected[role] > 0;
      const p = $(`peer-${role}`);
      p.classList.toggle("on", on);
      p.querySelector(".state").textContent = on ? "연결됨" : "대기";
    }
    $("rec-start").disabled = ev.recording || ev.status !== "created";
    $("rec-stop").disabled = !ev.recording;
    $("rec-status").classList.toggle("live", ev.recording);
    if (ev.recording && !R.timer) {
      const t0 = Date.now() - ev.elapsed_s * 1000;
      R.timer = setInterval(() => { $("rec-status").textContent = `● 녹음 중 ${fmtDur((Date.now() - t0) / 1000)}`; }, 250);
    } else if (!ev.recording) {
      stopTimer();
      $("rec-status").textContent = ev.status === "recorded" ? "녹음이 끝났습니다" : "녹음 전";
    }
    if (ev.mode === "turns") {
      const panel = $("turn-panel");
      panel.hidden = !ev.recording;
      if (ev.recording && ev.active) {
        const next = ev.active === "tutor" ? "student" : "tutor";
        panel.className = `turn-panel ${ev.active}`;
        $("turn-now").textContent = `지금: ${ROLE_KO[ev.active]} 차례 (${ev.turn_count}번째)`;
        $("turn-next").textContent = `${ROLE_KO[ev.active]} 완료 → ${ROLE_KO[next]} 차례`;
      }
    } else if (ev.recording && ev.connected.student === 0) {
      banner("학생이 연결되어 있지 않습니다. 학생 쪽은 무음으로 기록됩니다.");
    }
  } else if (ev.type === "recorded") {
    if (R.role === "student") { $("connect").hidden = true; } else { showResult(ev.meta); }
    closeMics();
  } else if (ev.type === "error") {
    banner(ev.detail);
  }
}

function showResult(m) {
  $("result").hidden = false;
  $("connect").hidden = true;
  $("rec-start").hidden = true;
  $("rec-stop").hidden = true;
  const dl = $("result-dl");
  dl.replaceChildren();
  const add = (k, v) => { const dt = document.createElement("dt"); dt.textContent = k; const dd = document.createElement("dd"); dd.textContent = v; dl.append(dt, dd); };
  add("길이", fmtDur(m.duration_s));
  for (const role of ["tutor", "student"]) {
    const t = (m.tracks || {})[role];
    if (t) add(ROLE_KO[role], `받은 음성 ${t.received_s}초, 늦게 들어와 채운 무음 ${t.lead_pad_s}초, 끊김 ${t.gap_pad_s}초, 시계 차이 ${t.drift_s}초`);
  }
  if (m.turns) {
    const n = (r) => m.turns.filter((t) => t.role === r).length;
    add("차례", `모두 ${m.turns.length}번 (튜터 ${n("tutor")}번, 학생 ${n("student")}번)`);
  }
  add("학습 활용 동의", m.training_consent ? "예" : "아니오 (녹음만 보관)");
  for (const role of ["tutor", "student"]) $(`play-${role}`).src = `/api/sessions/${m.id}/audio/${role}.wav`;
}

async function enterRoom() {
  R.sid = qs.get("sid");
  R.role = qs.get("role") || "tutor";
  R.roles = R.role === "both" ? ["tutor", "student"] : [R.role];  // turns 는 ["turns"]: 공용 마이크 하나
  show("room");
  if (R.role === "student") {
    // 학생 화면에서는 다른 세션 목록·재생·삭제로 가는 메뉴를 숨긴다
    $("nav").hidden = true;
    document.querySelector("h1 a").removeAttribute("href");
  }
  const r = await fetch(`/api/sessions/${R.sid}`);
  if (!r.ok) { banner("세션을 찾을 수 없습니다."); return; }
  R.meta = await r.json();
  $("room-title").textContent = { both: "녹음 (기기 1대)", turns: "순차 녹음 (튜터 → 학생 차례대로)" }[R.role] || `${ROLE_KO[R.role]} 화면`;
  if (R.role === "turns") document.querySelector(".peers").hidden = true;
  $("room-info").textContent = `세션 ${R.sid} · 튜터 ${R.meta.tutor_alias} · 학생 ${R.meta.student_alias}`
    + (R.meta.student_group === "child" ? ` (아동 ${R.meta.age_band})` : "") + (R.meta.topic ? ` · ${R.meta.topic}` : "");
  if (R.meta.status === "recorded") {
    if (R.role === "student") { $("rec-status").textContent = "녹음이 끝났습니다"; $("connect").hidden = true; return; }
    showResult(R.meta);
    return;
  }
  for (const role of R.roles) $("mics").append(micBlock(role));
  if (R.roles.length > 1) {
    $("listen-wrap").hidden = true;
    banner("같은 방에서 녹음하면 서로의 목소리가 상대 마이크에도 들어갑니다. 마이크를 각자 입 가까이 두세요.", "info");
  }
  if (R.role === "student") $("rec-status").textContent = "튜터가 녹음을 시작합니다";
  if (R.role === "turns") {
    $("listen-wrap").hidden = true;
    banner("튜터 차례부터 시작합니다. 말을 마칠 때마다 '튜터 완료', '학생 완료'를 누르세요. 겹쳐 말한 부분은 차례인 사람 쪽에 기록됩니다.", "info");
  }
  try { await listDevices(); } catch (_) { /* 권한 전에는 이름 없이 보임 */ }
  $("connect").onclick = connect;
  $("rec-start").onclick = () => send({ type: "start" });
  $("rec-stop").onclick = () => send({ type: "stop" });
  $("listen").onchange = () => { for (const ws of Object.values(R.socks)) if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "listen", on: $("listen").checked })); };
  $("turn-next").onclick = (e) => { e.currentTarget.blur(); send({ type: "turn" }); };
  document.addEventListener("keyup", (e) => {  // 버튼은 스페이스를 뗄 때 눌리므로 keyup 도 막는다
    if (e.code === "Space" && R.state && R.state.recording && R.state.mode === "turns"
        && !["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement && document.activeElement.tagName)) e.preventDefault();
  });
  document.addEventListener("keydown", (e) => {
    // 스페이스 키로 차례 바꾸기. 초점이 있는 버튼(예: 녹음 종료)이 스페이스로 눌리지 않게 기본 동작을 막는다
    if (e.code !== "Space" || e.repeat || !R.state || !R.state.recording || R.state.mode !== "turns") return;
    const tag = document.activeElement && document.activeElement.tagName;
    if (["INPUT", "SELECT", "TEXTAREA"].includes(tag)) return;
    e.preventDefault();
    send({ type: "turn" });
  });
  window.addEventListener("beforeunload", (e) => { if (R.state && R.state.recording && ctrlSock()) { e.preventDefault(); e.returnValue = ""; } });
}

// ---------- 시작 ----------
if (qs.get("sid")) {
  enterRoom();
} else {
  const view = qs.get("view") || "new";
  show(view);
  if (view === "new") setupForm();
  if (view === "list") loadList();
}
