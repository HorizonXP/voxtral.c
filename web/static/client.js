/* Voxtral realtime dictation demo (browser).
 *
 * Captures microphone audio, downsamples to 16kHz mono PCM16 in an AudioWorklet,
 * streams bytes to the server via WebSocket, and renders transcript deltas.
 */

let ws = null;
let audioCtx = null;
let mediaStream = null;
let sourceNode = null;
let workletNode = null;

const $ = (id) => document.getElementById(id);
const elServerUrl = $("serverUrl");
const elInterval = $("interval");
const elStart = $("btnStart");
const elStop = $("btnStop");
const elStatus = $("status");
const elTranscript = $("transcript");
const elCopy = $("btnCopy");
const elClear = $("btnClear");

function setStatus(s) {
  elStatus.textContent = s;
}

function appendText(t) {
  elTranscript.textContent += t;
  elTranscript.scrollTop = elTranscript.scrollHeight;
}

function clearText() {
  elTranscript.textContent = "";
}

function wsUrl() {
  const base = (elServerUrl.value || window.location.origin).trim();
  const u = new URL(base);
  u.protocol = (u.protocol === "https:") ? "wss:" : "ws:";
  u.pathname = "/v1/audio/transcriptions/realtime";
  const interval = (elInterval.value || "0.5").trim();
  u.searchParams.set("interval", interval);
  return u.toString();
}

async function start() {
  if (ws) return;

  setStatus("Requesting microphone...");
  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });

  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  await audioCtx.audioWorklet.addModule("/static/pcm16_downsample_worklet.js");

  sourceNode = audioCtx.createMediaStreamSource(mediaStream);
  workletNode = new AudioWorkletNode(audioCtx, "voxtral-pcm16-downsample");

  // Connect the graph (no audible output).
  sourceNode.connect(workletNode);
  workletNode.connect(audioCtx.destination);

  setStatus("Connecting...");
  ws = new WebSocket(wsUrl());
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    setStatus("Streaming");
    elStart.disabled = true;
    elStop.disabled = false;
  };

  ws.onmessage = (ev) => {
    // Server sends JSON text events.
    if (typeof ev.data === "string") {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === "transcript.delta" && msg.text) {
          appendText(msg.text);
        } else if (msg.type === "transcript.done") {
          setStatus("Done");
        } else if (msg.type === "error") {
          setStatus("Error");
          console.error("server error:", msg);
        }
      } catch (e) {
        console.warn("non-json message:", ev.data);
      }
      return;
    }
  };

  ws.onclose = () => {
    cleanup();
    setStatus("Idle");
  };

  ws.onerror = (e) => {
    console.error("ws error:", e);
    setStatus("Error");
  };

  // Send PCM chunks to server as binary frames.
  workletNode.port.onmessage = (ev) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (!(ev.data instanceof ArrayBuffer)) return;
    ws.send(ev.data);
  };
}

function cleanup() {
  if (ws) {
    try { ws.close(); } catch (_) {}
    ws = null;
  }

  if (workletNode) {
    try { workletNode.disconnect(); } catch (_) {}
    workletNode = null;
  }
  if (sourceNode) {
    try { sourceNode.disconnect(); } catch (_) {}
    sourceNode = null;
  }

  if (mediaStream) {
    for (const t of mediaStream.getTracks()) t.stop();
    mediaStream = null;
  }

  if (audioCtx) {
    try { audioCtx.close(); } catch (_) {}
    audioCtx = null;
  }

  elStart.disabled = false;
  elStop.disabled = true;
}

async function stop() {
  if (!ws) return cleanup();
  setStatus("Stopping...");
  try {
    ws.send(JSON.stringify({ type: "stop" }));
  } catch (_) {
    cleanup();
  }
}

elStart.addEventListener("click", () => start().catch((e) => {
  console.error(e);
  setStatus("Error");
  cleanup();
}));

elStop.addEventListener("click", () => stop());

elCopy.addEventListener("click", async () => {
  const t = elTranscript.textContent || "";
  if (!t) return;
  try {
    await navigator.clipboard.writeText(t);
  } catch (e) {
    console.warn("clipboard write failed:", e);
  }
});

elClear.addEventListener("click", () => clearText());

// Default server URL to current origin.
elServerUrl.value = window.location.origin;
setStatus("Idle");

