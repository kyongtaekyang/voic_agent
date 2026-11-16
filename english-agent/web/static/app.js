// English Agent screen. /ws carries binary PCM16 (24 kHz mono, little-endian) both ways plus JSON eva.* events
// (docs/event_schema.md). Moshi is full-duplex: the mic stays open while the tutor speaks.

const RATE = 24000;
const $ = (id) => document.getElementById(id);
const devRequested = new URLSearchParams(location.search).get("dev") === "1";

const S = {
  ws: null, stream: null, micCtx: null, micNode: null, playCtx: null,
  playHead: 0, sources: [], agentEl: null, lastDelta: 0, status: null, gpuTimer: null, koreanUrl: null,
};

const STATUS_TEXT = {
  idle: "Press Start to begin.",
  connecting: "Connecting…",
  listening: "I'm listening. Go ahead!",
  speaking: "Tutor is speaking…",
};

function setState(state) {
  S.status = state;
  $("orb").dataset.state = state === "connecting" ? "idle" : state;
  $("status").textContent = STATUS_TEXT[state];
}

function devLog(text) {
  if ($("dev").hidden) return;
  const li = document.createElement("li");
  li.textContent = `${new Date().toLocaleTimeString()} ${text}`;
  $("dev-events").prepend(li);
  while ($("dev-events").children.length > 60) $("dev-events").lastChild.remove();
}

function addLine(who, text) {
  const li = document.createElement("li");
  li.className = who;
  const label = document.createElement("span");
  label.className = "who";
  label.textContent = who === "user" ? "You" : "Tutor";
  const body = document.createElement("span");
  body.textContent = text || "";
  li.append(label, body);
  $("log").append(li);
  $("log").scrollTop = $("log").scrollHeight;
  return li;
}

// ---------- playback ----------
function enqueuePcm(buf) {
  if (!S.playCtx) return;
  const pcm = new Int16Array(buf);
  if (!pcm.length) return;
  const f = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) f[i] = pcm[i] / 0x8000;
  const ab = S.playCtx.createBuffer(1, f.length, RATE);
  ab.copyToChannel(f, 0);
  const src = S.playCtx.createBufferSource();
  src.buffer = ab;
  src.connect(S.playCtx.destination);
  const at = Math.max(S.playCtx.currentTime + 0.05, S.playHead);
  src.start(at);
  S.playHead = at + ab.duration;
  S.sources.push(src);
  src.onended = () => { S.sources = S.sources.filter((x) => x !== src); };
}

function stopPlayback() {
  for (const src of S.sources) { src.onended = null; try { src.stop(); } catch (_) { /* already ended */ } }
  S.sources = [];
  S.playHead = S.playCtx ? S.playCtx.currentTime : 0;
}

// ---------- events ----------
function applyStatus(st) {
  $("maintenance").hidden = !st.maintenance;
  $("start").disabled = !!st.maintenance || !!S.ws;
  S.koreanUrl = st.korean_agent_url || S.koreanUrl;
  if (st.developer_panel && devRequested) {
    $("dev").hidden = false;
    $("dev-mode").textContent = st.mode || "(none)";
    $("dev-delay").textContent = `${st.output_delay_ms ?? "-"} ms, ASR ${st.asr ? "on" : "off"}, redirect audio ${st.redirect_audio ? "yes" : "no"}`;
    $("dev-last").textContent = st.last_session ? JSON.stringify(st.last_session) : "-";
    if (!S.gpuTimer) { refreshGpu(); S.gpuTimer = setInterval(refreshGpu, 5000); }
  }
}

function onEvent(ev) {
  switch (ev.type) {
    case "eva.session.started":
      setState("listening");
      devLog("session started");
      break;
    case "eva.state":
      if (S.ws) setState(ev.state);
      break;
    case "eva.agent_text": {
      const now = performance.now();
      if (!S.agentEl || now - S.lastDelta > 1500) S.agentEl = addLine("agent", "");
      S.agentEl.lastChild.textContent = (S.agentEl.lastChild.textContent + ev.delta).replace(/^\s+/, "");
      S.lastDelta = now;
      $("log").scrollTop = $("log").scrollHeight;
      break;
    }
    case "eva.user_text":
      if (ev.text) { addLine("user", ev.text); S.agentEl = null; }
      devLog(`ASR ${ev.asr_ms} ms for ${ev.audio_s} s (lang ${ev.lang || "?"})`);
      break;
    case "eva.language_mismatch":
      $("language-link").href = ev.korean_agent_url || S.koreanUrl || "#";
      $("language").hidden = false;
      devLog(`language mismatch: ${ev.detected_language} x${ev.turns}`);
      break;
    case "eva.safety_block":
      stopPlayback(); // drop audio already queued in the browser too
      S.agentEl = addLine("agent", ev.redirect_text || "");
      devLog(`SAFETY BLOCK ${ev.category}/${ev.rule} dropped ${ev.dropped_ms} ms, already played ${ev.already_played_ms} ms`);
      break;
    case "eva.safety_resumed":
      S.agentEl = null;
      devLog("safety: output resumed");
      break;
    case "eva.unavailable": {
      const msg = { maintenance: "The tutor is currently under maintenance.", busy: "The tutor is talking with someone else. Please try again in a moment." };
      $("unavailable").textContent = msg[ev.reason] || "The tutor is not available right now.";
      $("unavailable").hidden = false;
      devLog(`unavailable: ${ev.reason} ${ev.detail || ""}`);
      stop(false);
      break;
    }
    case "eva.session.ended":
      devLog(`session ended: ${JSON.stringify(ev)}`);
      break;
    case "eva.error":
      devLog(`error (${ev.source}): ${ev.detail}`);
      break;
    default:
      break;
  }
}

async function refreshGpu() {
  try {
    const g = await (await fetch("/api/dev/gpu")).json();
    $("dev-gpu").textContent = g.error ? g.error : `${(g.used_mib / 1024).toFixed(1)} / ${(g.total_mib / 1024).toFixed(1)} GB (${g.name})`;
  } catch (_) { $("dev-gpu").textContent = "unavailable"; }
}

// ---------- start / stop ----------
async function start() {
  $("unavailable").hidden = true;
  $("start").disabled = true;
  setState("connecting");
  try {
    S.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
  } catch (e) {
    $("mic").textContent = "Microphone: not available (please allow microphone access)";
    setState("idle");
    $("start").disabled = false;
    return;
  }
  $("mic").textContent = "Microphone: on";
  S.playCtx = new AudioContext();
  S.micCtx = new AudioContext();
  await S.micCtx.audioWorklet.addModule("/static/capture-worklet.js");
  const src = S.micCtx.createMediaStreamSource(S.stream);
  S.micNode = new AudioWorkletNode(S.micCtx, "pcm16-capture", { processorOptions: { targetRate: RATE, chunkMs: 80 } });
  const mute = S.micCtx.createGain();
  mute.gain.value = 0;
  src.connect(S.micNode).connect(mute).connect(S.micCtx.destination);

  const proto = location.protocol === "https:" ? "wss" : "ws";
  S.ws = new WebSocket(`${proto}://${location.host}/ws`);
  S.ws.binaryType = "arraybuffer";
  S.ws.onopen = () => {
    S.micNode.port.onmessage = (e) => { if (S.ws && S.ws.readyState === WebSocket.OPEN) S.ws.send(e.data); };
  };
  S.ws.onmessage = (e) => {
    if (typeof e.data === "string") { try { onEvent(JSON.parse(e.data)); } catch (err) { console.error(err); } }
    else enqueuePcm(e.data);
  };
  S.ws.onclose = () => { if (S.ws) stop(false); };
  $("stop").disabled = false;
}

function stop(userAction = true) {
  const ws = S.ws;
  S.ws = null;
  if (ws && ws.readyState <= WebSocket.OPEN) ws.close();
  stopPlayback();
  if (S.micNode) { S.micNode.port.onmessage = null; S.micNode.disconnect(); S.micNode = null; }
  if (S.stream) { S.stream.getTracks().forEach((t) => t.stop()); S.stream = null; }
  if (S.micCtx) { S.micCtx.close(); S.micCtx = null; }
  if (S.playCtx) { S.playCtx.close(); S.playCtx = null; }
  S.agentEl = null;
  $("mic").textContent = "Microphone: off";
  $("stop").disabled = true;
  $("start").disabled = !$("maintenance").hidden;
  setState("idle");
  if (userAction) devLog("session stopped");
  setTimeout(() => fetch("/api/status").then((r) => r.json()).then(applyStatus).catch(() => {}), 2500);
}

$("start").addEventListener("click", start);
$("stop").addEventListener("click", () => stop(true));
$("language-close").addEventListener("click", () => { $("language").hidden = true; });

fetch("/api/status").then((r) => r.json()).then(applyStatus).catch(() => {});
setInterval(() => { if (!S.ws) fetch("/api/status").then((r) => r.json()).then(applyStatus).catch(() => {}); }, 15000);
setState("idle");
