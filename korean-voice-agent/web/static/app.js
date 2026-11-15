// 한국어 Agent 화면. 서버 /ws 는 Realtime 서버 이벤트를 그대로 넘기고 kva.* 이벤트를 덧붙인다 (docs/event_schema.md).
// 음성: 마이크 → 16kHz PCM16 → input_audio_buffer.append / 응답 response.output_audio.delta(16kHz PCM16) → 재생.

const RATE = 16000;
const SNAPSHOT_EXT = "speech_to_speech.input_audio_transcription.snapshot";
const $ = (id) => document.getElementById(id);
const devRequested = new URLSearchParams(location.search).get("dev") === "1";

const S = {
  ws: null, stream: null, micCtx: null, micNode: null, playCtx: null,
  playHead: 0, sources: [], playing: null, // playing = {itemId, startedAt(ctx time), ms}
  responseDone: true, userEl: null, botEl: null, gotSnapshot: false, status: null, gpuTimer: null,
};

const STATUS_TEXT = {
  idle: "대화를 시작하려면 아래 버튼을 눌러 주세요.",
  connecting: "연결하는 중입니다…",
  listening: "듣는 중이에요. 편하게 말해 주세요.",
  thinking: "생각하는 중이에요…",
  speaking: "말하는 중이에요.",
};

function setState(state) {
  S.status = state;
  $("orb").dataset.state = state === "connecting" ? "idle" : state;
  $("status").textContent = STATUS_TEXT[state];
}

function devLog(text) {
  if ($("dev").hidden) return;
  const li = document.createElement("li");
  li.textContent = `${new Date().toLocaleTimeString("ko-KR")} ${text}`;
  $("dev-events").prepend(li);
  while ($("dev-events").children.length > 40) $("dev-events").lastChild.remove();
}

function addLine(who) {
  const li = document.createElement("li");
  li.className = who;
  const label = document.createElement("span");
  label.className = "who";
  label.textContent = who === "user" ? "나" : "도우미";
  const body = document.createElement("span");
  li.append(label, body);
  $("log").append(li);
  $("log").scrollTop = $("log").scrollHeight;
  return li;
}

function setLine(li, text, partial) {
  li.lastChild.textContent = text;
  li.classList.toggle("partial", !!partial);
  $("log").scrollTop = $("log").scrollHeight;
}

function b64FromBuffer(buf) {
  const bytes = new Uint8Array(buf);
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

function pcm16FromB64(b64) {
  const bin = atob(b64);
  const out = new Float32Array(bin.length >> 1);
  for (let i = 0; i < out.length; i++) {
    let v = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
    if (v >= 0x8000) v -= 0x10000;
    out[i] = v / 0x8000;
  }
  return out;
}

function send(obj) {
  if (S.ws && S.ws.readyState === WebSocket.OPEN) S.ws.send(JSON.stringify(obj));
}

// ---------- 재생 ----------
function enqueueAudio(itemId, b64) {
  const samples = pcm16FromB64(b64);
  if (!samples.length) return;
  const ctx = S.playCtx;
  const buf = ctx.createBuffer(1, samples.length, RATE); // 컨텍스트가 장치 샘플레이트로 변환해 재생한다
  buf.copyToChannel(samples, 0);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  const at = Math.max(ctx.currentTime + 0.02, S.playHead);
  if (!S.playing || S.playing.itemId !== itemId) S.playing = { itemId, startedAt: at };
  src.start(at);
  S.playHead = at + buf.duration;
  S.sources.push(src);
  src.onended = () => {
    S.sources = S.sources.filter((x) => x !== src);
    if (!S.sources.length && S.responseDone) { S.playing = null; if (S.ws) setState("listening"); }
  };
  setState("speaking");
}

// 끼어들기: 남은 재생을 즉시 멈추고, 실제로 들려준 길이까지만 이력에 남도록 서버에 알린다.
function stopPlayback(reason) {
  if (!S.sources.length) return;
  const t0 = performance.now();
  for (const src of S.sources) { src.onended = null; try { src.stop(); } catch (_) { /* 이미 끝남 */ } }
  S.sources = [];
  if (S.playing && reason === "barge-in") {
    const ms = Math.max(0, Math.round((S.playCtx.currentTime - S.playing.startedAt) * 1000));
    send({ type: "conversation.item.truncate", item_id: S.playing.itemId, content_index: 0, audio_end_ms: ms });
    devLog(`끼어들기: 재생 중단 (${ms}ms 재생, 처리 ${(performance.now() - t0).toFixed(1)}ms)`);
  }
  S.playing = null;
  S.playHead = S.playCtx ? S.playCtx.currentTime : 0;
}

// ---------- 서버 이벤트 ----------
function onEvent(ev) {
  switch (ev.type) {
    case "kva.status":
      applyStatus(ev);
      break;
    case "kva.maintenance":
      showMaintenance(true);
      stop(false);
      break;
    case "kva.unavailable":
      $("unavailable-title").textContent = ev.reason === "busy"
        ? "다른 사람이 대화하고 있습니다." : "음성 서버에 연결하지 못했습니다.";
      $("unavailable").hidden = false;
      stop(false);
      break;
    case "kva.language_mismatch":
      $("language").hidden = false;
      break;
    case "kva.metrics":
      addLatencyRow(ev);
      break;
    case "kva.server_latency":
      addServerLatencyRow(ev);
      break;
    case "kva.vad":
      $("dev-vad").textContent = `발화 시작 감지 지연 ${ev.detect_delay_ms}ms`;
      break;
    case "kva.guard": {
      const li = document.createElement("li");
      li.textContent = `${new Date((ev.ts || Date.now() / 1000) * 1000).toLocaleTimeString("ko-KR")} ${ev.event === "guard_blocked" ? "차단" : "이력 정리"} · ${ev.rule} · "${ev.matched}"`;
      $("dev-guard").prepend(li);
      break;
    }
    case "session.created":
    case "session.updated":
      if (S.status !== "speaking") setState("listening");
      break;
    case "input_audio_buffer.speech_started":
      stopPlayback("barge-in");
      setState("listening");
      S.userEl = addLine("user");
      S.gotSnapshot = false;
      break;
    case SNAPSHOT_EXT:
      if (!S.userEl) S.userEl = addLine("user");
      S.gotSnapshot = true;
      setLine(S.userEl, ev.transcript, true);
      break;
    case "conversation.item.input_audio_transcription.delta":
      if (!S.gotSnapshot) {
        if (!S.userEl) S.userEl = addLine("user");
        setLine(S.userEl, S.userEl.lastChild.textContent + (ev.delta || ""), true);
      }
      break;
    case "conversation.item.input_audio_transcription.completed":
      if (!S.userEl) S.userEl = addLine("user");
      if (ev.transcript && ev.transcript.trim()) setLine(S.userEl, ev.transcript, false);
      else S.userEl.remove();
      S.userEl = null;
      break;
    case "input_audio_buffer.speech_stopped":
      setState("thinking");
      break;
    case "response.created":
      S.responseDone = false;
      S.botEl = null;
      break;
    case "response.output_audio_transcript.delta":
      if (!S.botEl) S.botEl = addLine("bot");
      setLine(S.botEl, S.botEl.lastChild.textContent + (ev.delta || ""), false);
      break;
    case "response.output_audio.delta":
      enqueueAudio(ev.item_id || "item", ev.delta);
      break;
    case "response.done":
      S.responseDone = true;
      S.botEl = null;
      if (!S.sources.length) setState("listening");
      break;
    case "error":
      devLog(`오류: ${(ev.error && (ev.error.message || ev.error.code)) || "알 수 없음"}`);
      break;
    default:
      break;
  }
  if (ev.type && !ev.type.startsWith("response.output_audio.delta") && ev.type !== "input_audio_buffer.append") {
    if (!["response.output_audio_transcript.delta", SNAPSHOT_EXT, "conversation.item.input_audio_transcription.delta"].includes(ev.type)) devLog(ev.type);
  }
}

const STATUS_KO = { completed: "완료", cancelled: "취소", incomplete: "중단", failed: "실패" };

function addRow(tbodyId, cells) {
  const tr = document.createElement("tr");
  for (const c of cells) { const td = document.createElement("td"); td.textContent = c ?? "-"; tr.append(td); }
  $(tbodyId).prepend(tr);
  while ($(tbodyId).children.length > 12) $(tbodyId).lastChild.remove();
}

function addLatencyRow(m) {
  addRow("dev-e2e", [m.turn, m.turn_decision_ms, m.e2e_ms, STATUS_KO[m.status] || m.status]);
}

function addServerLatencyRow(m) {
  addRow("dev-lat", [m.turn_id, m.vad_decision_ms, m.hold_ms, m.stt_ms, m.llm_ms, m.tts_ttfa_ms,
    STATUS_KO[m.status] || m.status]);
}

function applyStatus(st) {
  if (st.english_agent_url) $("english-link").href = st.english_agent_url;
  showMaintenance(!!st.maintenance);
  if (devRequested && st.dev_screen) {
    $("dev").hidden = false;
    $("dev-mode").textContent = st.mode ? (st.mode === "realtime" ? "실시간 모드" : st.mode) : "설정 안 됨";
    const m = st.models || {};
    $("dev-llm").textContent = `${m.llm} (${m.llm_precision || "-"})`;
    $("dev-stt").textContent = `${m.stt} (${m.stt_device})`;
    $("dev-tts").textContent = `${m.tts} (${m.tts_backend})`;
    if (!S.gpuTimer) { refreshGpu(); S.gpuTimer = setInterval(refreshGpu, 5000); }
  }
}

function showMaintenance(on) {
  $("maintenance").hidden = !on;
  $("start").disabled = on || !!S.ws;
}

async function refreshGpu() {
  try {
    const r = await fetch("/api/dev/gpu");
    const g = await r.json();
    $("dev-gpu").textContent = g.error ? g.error : `${(g.used_mib / 1024).toFixed(1)} / ${(g.total_mib / 1024).toFixed(1)} GB (${g.name})`;
  } catch (_) { $("dev-gpu").textContent = "조회 실패"; }
}

// ---------- 시작·종료 ----------
async function start() {
  $("unavailable").hidden = true;
  $("start").disabled = true;
  setState("connecting");
  try {
    S.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
  } catch (e) {
    $("mic").textContent = "마이크: 사용할 수 없음 (브라우저에서 마이크 권한을 허용해 주세요)";
    setState("idle");
    $("start").disabled = false;
    return;
  }
  $("mic").textContent = "마이크: 켜짐";
  S.playCtx = new AudioContext();
  S.micCtx = new AudioContext();
  await S.micCtx.audioWorklet.addModule("/static/capture-worklet.js");
  const src = S.micCtx.createMediaStreamSource(S.stream);
  S.micNode = new AudioWorkletNode(S.micCtx, "pcm16-capture", { processorOptions: { targetRate: RATE } });
  const mute = S.micCtx.createGain();
  mute.gain.value = 0;
  src.connect(S.micNode).connect(mute).connect(S.micCtx.destination);

  const proto = location.protocol === "https:" ? "wss" : "ws";
  S.ws = new WebSocket(`${proto}://${location.host}/ws`);
  S.ws.onopen = () => {
    send({ type: "session.update", session: { type: "realtime", extensions: [SNAPSHOT_EXT],
      audio: { input: { turn_detection: { type: "server_vad", interrupt_response: true } }, output: {} } } });
    S.micNode.port.onmessage = (e) => send({ type: "input_audio_buffer.append", audio: b64FromBuffer(e.data) });
  };
  S.ws.onmessage = (e) => { try { onEvent(JSON.parse(e.data)); } catch (err) { console.error(err); } };
  S.ws.onclose = () => { if (S.ws) stop(false); };
  $("stop").disabled = false;
}

function stop(userAction = true) {
  const ws = S.ws;
  S.ws = null;
  if (ws && ws.readyState <= WebSocket.OPEN) ws.close();
  stopPlayback("stop");
  if (S.micNode) { S.micNode.port.onmessage = null; S.micNode.disconnect(); S.micNode = null; }
  if (S.stream) { S.stream.getTracks().forEach((t) => t.stop()); S.stream = null; }
  if (S.micCtx) { S.micCtx.close(); S.micCtx = null; }
  if (S.playCtx) { S.playCtx.close(); S.playCtx = null; }
  $("mic").textContent = "마이크: 꺼짐";
  $("stop").disabled = true;
  $("start").disabled = !$("maintenance").hidden;
  setState("idle");
  if (userAction) devLog("대화 종료");
}

$("start").addEventListener("click", start);
$("stop").addEventListener("click", () => stop(true));
$("language-close").addEventListener("click", () => { $("language").hidden = true; });

fetch("/api/status").then((r) => r.json()).then(applyStatus).catch(() => {});
setState("idle");
