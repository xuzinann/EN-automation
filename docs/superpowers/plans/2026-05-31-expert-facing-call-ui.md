# Expert-Facing Live Call UI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second, self-contained page (`/expert`) where a remote expert takes the live call directly with the AI interviewer "Shaun" — speaking via mic, hearing the agent's TTS visualized as a waveform — with the agent opening and driving the conversation.

**Architecture:** The expert page is the *sole* live WebSocket client and reuses the existing per-connection call loop in `routes_call.py` (no fan-out, no shared runtime). Two backend additions: a **proactive opening** (the agent speaks `opening_script` + first question ~2 s after the expert joins) and **`GET /api/call/active`** so the page can auto-join the live session. The operator's `index.html` gets one additive button that sets the session live and opens `/expert`.

**Tech Stack:** FastAPI + WebSockets (existing), Vertex AI via ADC (Gemini/Claude/STT/TTS — unchanged), vanilla JS + WebAudio API for the new page.

**Spec:** `docs/superpowers/specs/2026-05-31-expert-facing-call-ui-design.md`

**Project conventions (read before starting):**
- Work **directly on `main`** — no feature branches.
- **Hackathon testing posture:** skip heavy automated tests; verify with the lightweight checks each task specifies (syntax/import smoke, a tiny SessionManager check, `node --check`, and a final manual smoke over the IAP tunnel). The app runs on `127.0.0.1:8899` (NOT 8888 — another user's copy on this shared VM).
- `python3` only (`python` is not on PATH). All commands assume cwd `/home/kevin/EN-automation/expert_call_agent` unless stated.
- Commit message trailer (every commit): `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `expert_call_agent/config.py` | Opening-delay knob | Modify (append 1 const) |
| `expert_call_agent/session.py` | Track the live session for `/active` | Modify (`SessionManager`) |
| `expert_call_agent/api/routes_call.py` | `speak()` helper, proactive opening, `/active` | Modify |
| `expert_call_agent/api/app.py` | Serve `/expert` | Modify (1 route) |
| `expert_call_agent/static/expert.html` | The expert page (UI + WS + mic + waveform) | **Create** |
| `expert_call_agent/static/index.html` | "Start Expert Call" button (additive) | Modify |

---

## Task 1: Add the opening-delay config knob

**Files:**
- Modify: `expert_call_agent/config.py` (append at end, after line 81)

- [ ] **Step 1: Append the constant**

Add to the very end of `config.py`:

```python

# Seconds to wait after the expert joins a live call before the agent speaks its
# opening turn — lets the page render, the audio context unlock, and the mic spin
# up so the greeting isn't clipped. Set to 0 to disable the delay.
LIVE_OPENING_DELAY_SECONDS = 2.0
```

- [ ] **Step 2: Verify it loads**

Run: `python3 -c "import config; print(config.LIVE_OPENING_DELAY_SECONDS)"`
Expected: `2.0`

- [ ] **Step 3: Commit**

```bash
git add expert_call_agent/config.py
git commit -m "config: add LIVE_OPENING_DELAY_SECONDS for the expert-call opening

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Track the live session in SessionManager

**Files:**
- Modify: `expert_call_agent/session.py`

The expert page needs to discover which session is live. We track a single `_live_session_id` (the demo assumes one live call at a time), set when a session transitions to `status == "live"` and cleared when it leaves.

- [ ] **Step 1: Initialize the field**

In `session.py`, in `SessionManager.__init__` (currently lines 6-9), add `_live_session_id`:

```python
    def __init__(self):
        self._sessions: dict[str, CallSession] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._active_calls: set[str] = set()
        self._live_session_id: str | None = None
```

- [ ] **Step 2: Maintain it inside `update_session`**

Replace the existing `update_session` (currently lines 26-31) with:

```python
    async def update_session(self, session_id: str, **updates) -> CallSession:
        async with self._get_lock(session_id):
            session = self._sessions[session_id]
            for key, value in updates.items():
                setattr(session, key, value)
            if "status" in updates:
                if updates["status"] == "live":
                    self._live_session_id = session_id
                elif self._live_session_id == session_id:
                    self._live_session_id = None
            return session
```

- [ ] **Step 3: Add the getter**

Add this method to `SessionManager` (e.g. directly after `update_session`):

```python
    def get_live_session_id(self) -> str | None:
        """The session_id of the current live call, or None.

        Validates against live status so a stale pointer (e.g. a session that
        ended without a status update) is never returned.
        """
        sid = self._live_session_id
        session = self._sessions.get(sid) if sid else None
        if session is not None and session.status == "live":
            return sid
        return None
```

- [ ] **Step 4: Verify with a focused check**

Run:
```bash
python3 -c "
import asyncio
from session import SessionManager
async def main():
    m = SessionManager()
    s = await m.create_session()
    assert m.get_live_session_id() is None
    await m.update_session(s.session_id, status='live')
    assert m.get_live_session_id() == s.session_id, 'should be live'
    await m.update_session(s.session_id, status='post_call')
    assert m.get_live_session_id() is None, 'should clear on leave'
    print('OK')
asyncio.run(main())
"
```
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add expert_call_agent/session.py
git commit -m "session: track the live session id for /api/call/active

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Extract a reusable `speak()` helper in routes_call.py

**Files:**
- Modify: `expert_call_agent/api/routes_call.py`

Pure refactor — no behavior change. The "emit one agent turn" tail of `process_entry` becomes a module-level helper so the proactive opening (Task 4) can reuse it.

- [ ] **Step 1: Add the helper function**

In `routes_call.py`, immediately after the `_run_agent` function (currently ends at line 38), add:

```python


async def speak(ws, app, session_id: str, utterance: str, *, agent_name: str,
                flag_type: str, rationale: str) -> None:
    """Emit one agent turn: ai_turn event, interviewer transcript entry, TTS audio.

    Shared by the reactive turn loop (process_entry) and the proactive opening so
    both produce an identical client-facing sequence.
    """
    sessions = app.state.sessions
    await ws.send_json({
        "type": "ai_turn",
        "question": utterance,
        "agent": agent_name,
        "flag_type": flag_type,
        "rationale": rationale,
    })

    ai_entry = TranscriptEntry(speaker="interviewer", text=utterance)
    await sessions.add_transcript_entry(session_id, ai_entry)
    await ws.send_json({"type": "transcript", "entry": ai_entry.model_dump()})

    try:
        audio_bytes = await app.state.tts.synthesize(utterance)
        await ws.send_json({
            "type": "tts_audio",
            "data": base64.b64encode(audio_bytes).decode(),
        })
    except Exception:
        pass
```

- [ ] **Step 2: Replace the inline tail in `process_entry`**

In `process_entry`, replace the block currently at lines 150-170 (from `await ws.send_json({` for the `ai_turn` through the `except Exception: pass` of the TTS block) with this single call:

```python
            await speak(
                ws, app, session_id, utterance,
                agent_name=selected.agent_name,
                flag_type=selected.flag_type,
                rationale=selected.metadata.get("rationale")
                or selected.metadata.get("reason", ""),
            )
```

The lines immediately above it are unchanged:
```python
            if not utterance or not utterance.strip():
                return
            utterance = utterance.strip()
```

- [ ] **Step 3: Syntax check**

Run: `python3 -c "import ast; ast.parse(open('api/routes_call.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Manual smoke (regression — demo still speaks)**

Start the server (`python3 run.py`, serves `127.0.0.1:8899`), open the UI over the tunnel, Use Sample Brief → Generate Call Guide → Run Demo Simulation. Confirm the AI still produces spoken turns (ai_turn cards + audio) exactly as before. Stop the server.

Expected: identical behavior to before the refactor (the helper is byte-for-byte the same sequence).

- [ ] **Step 5: Commit**

```bash
git add expert_call_agent/api/routes_call.py
git commit -m "routes_call: extract speak() helper from process_entry (no behavior change)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Proactive opening — the agent greets first

**Files:**
- Modify: `expert_call_agent/api/routes_call.py`

When the expert connects to a live session that has not been opened yet, the agent waits `LIVE_OPENING_DELAY_SECONDS` then speaks `opening_script` + the first guide question.

- [ ] **Step 1: Add the opening composer**

In `routes_call.py`, add this module-level helper directly after the `speak()` function from Task 3:

```python


def _compose_opening(session) -> str:
    """Build the agent's first spoken turn from the call guide.

    opening_script (the greeting, already phrased for 'Shaun') followed by the
    first available question: opening_questions -> first deep-dive question.
    Returns "" if the guide has nothing to say (opening is then skipped).
    """
    guide = session.call_guide
    if guide is None:
        return ""

    parts: list[str] = []
    if guide.opening_script:
        parts.append(guide.opening_script.strip())

    first_q = ""
    if guide.opening_questions:
        first_q = (guide.opening_questions[0].text or "").strip()
    else:
        for sec in guide.deep_dive_sections:
            if sec.questions:
                first_q = (sec.questions[0].text or "").strip()
                break
    if first_q:
        parts.append(first_q)

    return " ".join(p for p in parts if p).strip()
```

- [ ] **Step 2: Run the opening when the expert connects**

In `call_websocket`, locate the line that seeds the note watermark (currently line 90):

```python
        # On reconnect, don't re-extract notes for transcript entries already recorded.
        note_taker.seed_processed(len(session.transcript))
```

Immediately **after** that line (and before the `async def process_entry` definition), insert:

```python

        # Proactive opening: if this is a fresh live call (no interviewer turn yet),
        # the agent greets and asks the first question before the expert speaks.
        # The "no interviewer turn" guard is idempotent across reconnects.
        session = await sessions.get_session(session_id)
        already_opened = any(e.speaker == "interviewer" for e in session.transcript)
        if session.status == "live" and session.call_guide and not already_opened:
            opening = _compose_opening(session)
            if opening:
                await asyncio.sleep(config.LIVE_OPENING_DELAY_SECONDS)
                await speak(
                    ws, app, session_id, opening,
                    agent_name="orchestrator",
                    flag_type="opening",
                    rationale="Opening the call",
                )
```

- [ ] **Step 3: Syntax check**

Run: `python3 -c "import ast; ast.parse(open('api/routes_call.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Verify the composer in isolation**

Run:
```bash
python3 -c "
from models import CallGuide, Question
from api.routes_call import _compose_opening
class S:  # minimal stand-in for a session
    call_guide = CallGuide(opening_script='Hi, I am Shaun. Thanks for joining.',
                           opening_questions=[Question(text='Can you start with your role?')])
print(repr(_compose_opening(S())))
"
```
Expected: `'Hi, I am Shaun. Thanks for joining. Can you start with your role?'`

(If the import fails on path, run instead from the parent dir with `PYTHONPATH=expert_call_agent`.)

- [ ] **Step 5: Commit**

```bash
git add expert_call_agent/api/routes_call.py
git commit -m "routes_call: agent opens the call proactively after a settle delay

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `GET /api/call/active` endpoint

**Files:**
- Modify: `expert_call_agent/api/routes_call.py`

- [ ] **Step 1: Add the route**

In `routes_call.py`, add this endpoint after `end_call` (currently ends at line 55) and before the `@router.websocket` handler:

```python


@router.get("/active")
async def active_call(request: Request):
    """The live session id for the expert page to auto-join (None if no live call)."""
    sessions = request.app.state.sessions
    return {"session_id": sessions.get_live_session_id()}
```

(`Request` is already imported at the top of the file.)

- [ ] **Step 2: Syntax check**

Run: `python3 -c "import ast; ast.parse(open('api/routes_call.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Manual smoke**

Start the server. With `curl` over the tunnel (or on the VM): `curl -s http://127.0.0.1:8899/api/call/active` → `{"session_id":null}` before any call. After Use Sample Brief → Generate Guide → (click "Start Expert Call" once Task 8 lands, or `POST /api/call/start/<id>`), the same endpoint returns the live id. Stop the server.

Expected: `null` before a live call, the session id after.

- [ ] **Step 4: Commit**

```bash
git add expert_call_agent/api/routes_call.py
git commit -m "routes_call: add GET /api/call/active for expert auto-join

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Serve the expert page at `/expert`

**Files:**
- Modify: `expert_call_agent/api/app.py`

The catch-all `StaticFiles` mount at `/` is registered last; add an explicit `/expert` route **before** it so the clean URL resolves to `static/expert.html`.

- [ ] **Step 1: Import FileResponse**

At the top of `app.py`, after `from fastapi.staticfiles import StaticFiles` (line 7), add:

```python
from fastapi.responses import FileResponse
```

- [ ] **Step 2: Register the route before the `/` mount**

In `create_app`, the mounts are currently lines 40-43. Replace that block with (the `/expert` route is inserted *before* the `/` mount):

```python
    base_dir = Path(__file__).resolve().parent.parent
    app.mount("/sample_data", StaticFiles(directory=str(base_dir / "sample_data")), name="sample_data")
    app.mount("/context", StaticFiles(directory=str(base_dir / "context")), name="context")

    async def expert_page():
        return FileResponse(str(base_dir / "static" / "expert.html"))

    app.add_api_route("/expert", expert_page, methods=["GET"], include_in_schema=False)

    app.mount("/", StaticFiles(directory=str(base_dir / "static"), html=True), name="static")
```

- [ ] **Step 3: Syntax check**

Run: `python3 -c "import ast; ast.parse(open('api/app.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add expert_call_agent/api/app.py
git commit -m "app: serve the expert page at /expert

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

(Smoke for this route happens in Task 7's verification, once the file exists.)

---

## Task 7: Build the expert page — `static/expert.html`

**Files:**
- Create: `expert_call_agent/static/expert.html`

Self-contained page: a "Join call" ready screen (the click unlocks audio + mic — required by browser autoplay/permission policies), then the split-pane call view (dark waveform stage + light interleaved transcript), auto-joining the live session.

- [ ] **Step 1: Create the file**

Create `expert_call_agent/static/expert.html` with exactly this content:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Expert Call — Live</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  html,body{height:100%}
  body{font-family:-apple-system,'Segoe UI',Roboto,sans-serif;background:#070a14;overflow:hidden}
  .app{height:100vh;display:flex;flex-direction:column}

  /* ready screen */
  #ready{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;
    background:radial-gradient(circle at 50% 40%,#16203f,#070a14);color:#eaf1ff;text-align:center;padding:24px}
  #ready h1{font-size:22px;font-weight:600;margin:26px 0 8px}
  #ready p{color:#9fb3d1;font-size:14px;margin-bottom:24px;max-width:430px;line-height:1.5}
  #join-btn{background:#2e7d32;color:#fff;border:none;border-radius:10px;padding:13px 30px;font-size:15px;font-weight:600;cursor:pointer}
  #join-btn:hover{background:#1b5e20}
  #join-note{margin-top:16px;font-size:12px;color:#6f86ad;min-height:16px}

  /* call screen */
  #call{flex:1;display:none;flex-direction:column;min-height:0}
  .stage{flex:0 0 54%;position:relative;display:flex;flex-direction:column;align-items:center;
    justify-content:center;background:radial-gradient(circle at 50% 42%,#16203f,#080b16)}
  .topbar{position:absolute;top:0;left:0;right:0;display:flex;justify-content:space-between;
    align-items:center;padding:14px 18px;color:#9fb3d1;font-size:12px;z-index:5}
  .live{display:flex;align-items:center;gap:6px;color:#ff6b6b;font-weight:700;letter-spacing:.6px}
  .live .dot{width:8px;height:8px;border-radius:50%;background:#ff5252;box-shadow:0 0 8px #ff5252;animation:blink 1.4s infinite}
  @keyframes blink{50%{opacity:.25}}

  .wavewrap{position:absolute;left:0;right:0;top:48%;transform:translateY(-50%);height:170px;overflow:hidden;z-index:1}
  .wavewrap svg{width:220%;height:100%;display:block}
  .wave-amp{transform-box:fill-box;transform-origin:50% 50%;transform:scaleY(.4);transition:transform .09s ease-out}
  .wave-scroll{animation:wavescroll 1.9s linear infinite}
  @keyframes wavescroll{to{transform:translateX(-336px)}}

  .orb{width:88px;height:88px;border-radius:50%;
    background:radial-gradient(circle at 35% 30%,#9af7ea,#4aa3ff 55%,#6a5cff);
    box-shadow:0 0 40px rgba(90,160,255,.8),inset 0 0 14px rgba(154,247,234,.85);
    animation:breathe 3.2s ease-in-out infinite}
  @keyframes breathe{50%{transform:scale(1.06)}}
  .stage .orb{position:relative;z-index:4}
  .agentname{margin-top:20px;color:#eaf1ff;font-size:16px;font-weight:600;z-index:4}
  .state{margin-top:5px;color:#7fa6d8;font-size:12px;z-index:4;letter-spacing:.4px;min-height:16px}

  .scriptpane{flex:1;background:#f5f7fb;display:flex;flex-direction:column;min-height:0}
  .scripthead{padding:10px 18px;font-size:11px;text-transform:uppercase;letter-spacing:1.2px;
    color:#64748b;background:#eef2f7;border-bottom:1px solid #e2e8f0;display:flex;justify-content:space-between;align-items:center}
  .scriptbody{padding:16px 18px;overflow-y:auto;display:flex;flex-direction:column;gap:12px;flex:1}
  .msg{max-width:82%}
  .msg .who{font-size:10px;text-transform:uppercase;letter-spacing:.6px;margin-bottom:4px}
  .msg.agent{align-self:flex-start}
  .msg.agent .who{color:#2e7d32}
  .msg.agent .bubble{background:#fff;color:#1a2740;border:1px solid #dbe3ef;
    border-radius:12px 12px 12px 4px;padding:11px 14px;font-size:14px;line-height:1.55;box-shadow:0 1px 2px rgba(20,40,80,.05)}
  .msg.expert{align-self:flex-end;text-align:right}
  .msg.expert .who{color:#0d47a1}
  .msg.expert .bubble{background:#e3eefc;color:#13233f;border-radius:12px 12px 4px 12px;
    padding:9px 13px;font-size:13px;line-height:1.5;display:inline-block;text-align:left}

  .controls{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:11px 18px;
    background:#eef2f7;border-top:1px solid #e2e8f0}
  .mic{display:flex;align-items:center;gap:8px;color:#1b7a4b;font-size:12px;font-weight:600}
  .mic.off{color:#94a3b8}
  .micdot{width:9px;height:9px;border-radius:50%;background:#22b573;box-shadow:0 0 8px #22b573;animation:blink 1.6s infinite}
  .mic.off .micdot{background:#cbd5e1;box-shadow:none;animation:none}
  .text-fallback{font-size:11px;color:#94a3b8;flex:1;text-align:center}
  .text-fallback input{padding:6px 10px;border:1px solid #cbd5e1;border-radius:6px;font-size:12px;margin-left:6px;width:240px}
  .leave{background:#fde8ec;color:#c62828;border:1px solid #f3c2cb;border-radius:8px;
    padding:7px 14px;font-size:12px;font-weight:600;cursor:pointer}
</style>
</head>
<body>
<div class="app">
  <div id="ready">
    <div class="orb"></div>
    <h1>Shaun is ready to begin</h1>
    <p>You're about to join a live call with our AI interviewer. Click below to enable your microphone — Shaun will greet you and lead the conversation.</p>
    <button id="join-btn">Join call</button>
    <div id="join-note"></div>
  </div>

  <div id="call">
    <div class="stage">
      <div class="topbar">
        <span class="live"><span class="dot"></span>LIVE</span>
        <span id="call-label">Expert Call</span>
      </div>
      <div class="wavewrap">
        <svg viewBox="0 0 672 170" preserveAspectRatio="none">
          <g class="wave-amp"><g class="wave-scroll" fill="none" stroke-linecap="round">
            <path d="M-10 85 q 21 -38 42 0 t 42 0 t 42 0 t 42 0 t 42 0 t 42 0 t 42 0 t 42 0 t 42 0 t 42 0 t 42 0 t 42 0 t 42 0 t 42 0 t 42 0 t 42 0" stroke="#7ec8ff" stroke-width="2.5" opacity="0.9"/>
            <path d="M-10 85 q 21 38 42 0 t 42 0 t 42 0 t 42 0 t 42 0 t 42 0 t 42 0 t 42 0 t 42 0 t 42 0 t 42 0 t 42 0 t 42 0 t 42 0 t 42 0 t 42 0" stroke="#9af7ea" stroke-width="2" opacity="0.4"/>
          </g></g>
        </svg>
      </div>
      <div class="orb"></div>
      <div class="agentname">Shaun</div>
      <div class="state" id="state">Connecting…</div>
    </div>

    <div class="scriptpane">
      <div class="scripthead"><span>Transcript</span><span style="color:#94a3b8;">live</span></div>
      <div class="scriptbody" id="transcript"></div>
    </div>

    <div class="controls">
      <span class="mic off" id="mic"><span class="micdot"></span><span id="mic-text">Mic off</span></span>
      <span class="text-fallback">Trouble with audio?
        <input type="text" id="text-input" placeholder="Type an answer + Enter">
      </span>
      <button class="leave" id="leave-btn">Leave call</button>
    </div>
  </div>
</div>

<script>
let ws = null, sessionId = null, micRecorder = null, micStream = null;
let audioCtx = null, micActive = false, waveRaf = null;

// VAD (voice-activity detection) utterance endpointing — kept in sync with the
// canonical implementation in index.html. One complete WebM blob per utterance:
// MediaRecorder.start() with NO timeslice yields a single decodable blob on stop(),
// and a poll loop cuts the segment after a trailing pause. (Fixed-interval chunking
// produced headerless 2nd+ fragments the backend's STT couldn't decode — that bug
// is the whole reason this endpointer exists.)
const VAD_POLL_MS = 50;             // energy sampling cadence
const VAD_SILENCE_RMS = 0.015;      // RMS at/under this counts as silence
const VAD_SILENCE_MS = 800;         // trailing silence that ends an utterance
const VAD_MIN_UTTERANCE_MS = 400;   // drop blips shorter than this
const VAD_MAX_UTTERANCE_MS = 20000; // force-cut a long monologue
let micAnalyser = null, vadInterval = null, vadBuf = null;
let segmentStart = 0, lastVoice = 0, sawSpeech = false, segmentValid = false;

const forcedSession = new URLSearchParams(location.search).get('session');
const $ = id => document.getElementById(id);

function setState(t){ $('state').textContent = t; }

$('join-btn').addEventListener('click', joinCall);
$('leave-btn').addEventListener('click', leaveCall);
$('text-input').addEventListener('keydown', e => {
  const v = e.target.value.trim();
  if (e.key === 'Enter' && v && ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'text_input', speaker: 'expert', text: v }));
    e.target.value = '';
  }
});

async function joinCall(){
  // User gesture: unlock audio playback and request the mic.
  try { audioCtx = new (window.AudioContext || window.webkitAudioContext)(); await audioCtx.resume(); } catch(e) {}
  $('ready').style.display = 'none';
  $('call').style.display = 'flex';
  setState('Connecting…');
  try { await startMic(); } catch(e) { setMic(false, 'No mic — type answers below'); }
  const sid = await resolveSession();
  if (!sid) { setState('Waiting for the call to start…'); return; }
  sessionId = sid;
  connectWS();
}

async function resolveSession(){
  if (forcedSession) return forcedSession;
  for (let i = 0; i < 90; i++) {            // poll up to ~90s
    try {
      const r = await fetch('/api/call/active');
      const d = await r.json();
      if (d.session_id) return d.session_id;
    } catch(e) {}
    setState('Waiting for the call to start…');
    await new Promise(res => setTimeout(res, 1000));
  }
  return null;
}

function connectWS(){
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/api/call/ws/${sessionId}`);
  ws.onopen = () => setState(micActive ? 'Listening…' : 'Connected');
  ws.onmessage = e => handleMsg(JSON.parse(e.data));
  ws.onclose = () => { setState('Disconnected'); stopMic(); };
}

function handleMsg(msg){
  switch (msg.type) {
    case 'transcript': addTranscript(msg.entry); break;
    case 'ai_turn': setState('Speaking…'); break;
    case 'tts_audio': playAudio(msg.data); break;
    case 'error': console.error('agent error:', msg.message); break;
    // note, coverage_update, demo_progress, demo_complete: ignored on the expert page
  }
}

function addTranscript(entry){
  if (entry.speaker !== 'interviewer' && entry.speaker !== 'expert') return;
  const isAgent = entry.speaker === 'interviewer';
  const div = document.createElement('div');
  div.className = 'msg ' + (isAgent ? 'agent' : 'expert');
  div.innerHTML = `<div class="who">${isAgent ? 'Shaun · Interviewer' : 'You'}</div><div class="bubble"></div>`;
  div.querySelector('.bubble').textContent = entry.text;   // textContent = no HTML injection
  const box = $('transcript');
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
  if (!isAgent) setState('Thinking…');
}

function playAudio(b64){
  setState('Speaking…');
  const audio = new Audio('data:audio/mp3;base64,' + b64);
  try {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === 'suspended') audioCtx.resume();
    const src = audioCtx.createMediaElementSource(audio);
    const an = audioCtx.createAnalyser();
    an.fftSize = 256;
    src.connect(an);
    an.connect(audioCtx.destination);
    driveWave(an);
  } catch(e) { /* analyser optional; element still plays below */ }
  audio.onended = () => { stopWave(); setState(micActive ? 'Listening…' : 'Idle'); };
  audio.play().catch(err => console.log('audio blocked:', err));
}

function driveWave(an){
  if (waveRaf) cancelAnimationFrame(waveRaf);   // never stack loops across turns
  const ampEl = document.querySelector('.wave-amp');
  const data = new Uint8Array(an.frequencyBinCount);
  (function tick(){
    an.getByteFrequencyData(data);
    let sum = 0;
    for (let i = 0; i < data.length; i++) sum += data[i];
    const avg = sum / data.length;                 // 0..255
    const scale = 0.35 + Math.min(1.4, avg / 110); // idle floor + reactive amplitude
    if (ampEl) ampEl.style.transform = 'scaleY(' + scale.toFixed(3) + ')';
    waveRaf = requestAnimationFrame(tick);
  })();
}

function stopWave(){
  if (waveRaf) { cancelAnimationFrame(waveRaf); waveRaf = null; }
  const ampEl = document.querySelector('.wave-amp');
  if (ampEl) ampEl.style.transform = 'scaleY(0.4)';
}

async function startMic(){
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 16000 }
  });

  // Mic-energy analyser drives the VAD endpointer. Reuse the shared audioCtx
  // (already created + resumed by the Join gesture); the mic source feeds ONLY the
  // analyser — never audioCtx.destination — so the expert never hears themselves.
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === 'suspended') await audioCtx.resume();
  const source = audioCtx.createMediaStreamSource(micStream);
  micAnalyser = audioCtx.createAnalyser();
  micAnalyser.fftSize = 256;
  source.connect(micAnalyser);
  vadBuf = new Uint8Array(micAnalyser.fftSize);

  const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : 'audio/webm';
  micRecorder = new MediaRecorder(micStream, { mimeType: mime });
  // One segment per utterance: stop() flushes a single complete WebM blob;
  // segmentValid gates whether the blob is worth sending.
  micRecorder.ondataavailable = ev => {
    if (segmentValid && ev.data.size > 0 && ws && ws.readyState === WebSocket.OPEN) ws.send(ev.data);
  };
  // After each cut, immediately begin the next segment (unless the mic is off).
  micRecorder.onstop = () => { if (micActive && micRecorder) startSegment(); };

  startSegment();
  vadInterval = setInterval(vadTick, VAD_POLL_MS);
  setMic(true, 'Mic on — Shaun hears you');
}

// Begin recording a fresh utterance segment and reset its VAD state.
function startSegment(){
  segmentStart = performance.now();
  lastVoice = segmentStart;
  sawSpeech = false;
  segmentValid = false;
  if (micRecorder && micRecorder.state === 'inactive') micRecorder.start(); // no timeslice → one blob at stop()
}

// End the current segment; ondataavailable decides whether to actually send it.
function endSegment(){
  const dur = performance.now() - segmentStart;
  segmentValid = sawSpeech && dur >= VAD_MIN_UTTERANCE_MS;
  if (micRecorder && micRecorder.state === 'recording') micRecorder.stop(); // → ondataavailable → onstop → startSegment()
}

// Poll mic energy; cut the segment on a trailing pause or the max-length cap.
function vadTick(){
  if (!micAnalyser || !micRecorder || micRecorder.state !== 'recording') return;
  micAnalyser.getByteTimeDomainData(vadBuf);
  let sum = 0;
  for (let i = 0; i < vadBuf.length; i++){ const v = (vadBuf[i] - 128) / 128; sum += v * v; }
  const rms = Math.sqrt(sum / vadBuf.length);
  const now = performance.now();
  if (rms >= VAD_SILENCE_RMS){ sawSpeech = true; lastVoice = now; }
  if (sawSpeech && now - lastVoice >= VAD_SILENCE_MS) endSegment();
  else if (now - segmentStart >= VAD_MAX_UTTERANCE_MS) endSegment();
}

// Tear down the mic + VAD without touching the WS or the state line.
function stopMic(){
  micActive = false; // set first so onstop won't restart a segment
  if (vadInterval){ clearInterval(vadInterval); vadInterval = null; }
  if (micRecorder && micRecorder.state !== 'inactive'){
    // Flush a final in-progress utterance if it contained speech.
    segmentValid = sawSpeech && performance.now() - segmentStart >= VAD_MIN_UTTERANCE_MS;
    micRecorder.stop();
  }
  if (micStream){ micStream.getTracks().forEach(t => t.stop()); micStream = null; }
  micRecorder = null;
  micAnalyser = null;
  vadBuf = null;
  sawSpeech = false;
  setMic(false, 'Mic off');
}

function setMic(on, text){
  micActive = on;
  $('mic').className = 'mic' + (on ? '' : ' off');
  $('mic-text').textContent = text;
}

function leaveCall(){
  stopMic();
  if (ws) { ws.close(); ws = null; }
  setState('Call ended');
  $('text-input').disabled = true;
}
</script>
</body>
</html>
```

- [ ] **Step 2: JS syntax check**

Run:
```bash
sed -n '/<script>/,/<\/script>/p' static/expert.html | sed '1d;$d' > /tmp/expert.js && node --check /tmp/expert.js && echo OK
```
Expected: `OK`

- [ ] **Step 3: Route + page smoke**

Start the server. Over the tunnel, open `http://localhost:8899/expert` → the dark "Shaun is ready to begin" ready screen renders (confirms Task 6's `/expert` route serves the file). Don't click Join yet. Stop the server.

Expected: ready screen visible, no console errors (a favicon 404 is fine).

- [ ] **Step 4: Commit**

```bash
git add expert_call_agent/static/expert.html
git commit -m "static: add the expert-facing live call page (/expert)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: "Start Expert Call" button on the operator console

**Files:**
- Modify: `expert_call_agent/static/index.html`

Additive only — existing Start Call / Run Demo / mic controls are untouched. The new button sets the session live and opens `/expert` in a new tab. It does **not** open a WebSocket, so it won't collide with the expert's single live connection.

- [ ] **Step 1: Add the button**

In `index.html`, in the Phase 2 controls row (currently lines 220-225), add the new button after the existing `btn-demo` button. The block becomes:

```html
  <div class="controls">
    <button class="btn-success" id="btn-start-call" onclick="startCall()" disabled>Start Call</button>
    <button class="btn-danger" id="btn-end-call" onclick="endCall()" disabled>End Call</button>
    <button class="btn-outline" id="btn-demo" onclick="runDemo()" disabled>Run Demo Simulation</button>
    <button class="btn-primary" id="btn-expert-call" onclick="startExpertCall()" disabled>Start Expert Call</button>
    <span id="demo-progress"></span>
  </div>
```

- [ ] **Step 2: Enable it when the guide is ready**

In `index.html`, in `generateGuide()`, find the success lines (currently lines 473-474):

```javascript
    document.getElementById('btn-start-call').disabled = false;
    document.getElementById('btn-demo').disabled = false;
```

Add directly below them:

```javascript
    document.getElementById('btn-expert-call').disabled = false;
```

- [ ] **Step 3: Add the handler**

In `index.html`, add this function directly after `startCall()` (which currently ends at line 583):

```javascript

async function startExpertCall() {
  if (!sessionId) { alert('Generate a call guide first'); return; }
  try {
    await fetch(`/api/call/start/${sessionId}`, { method: 'POST' });
  } catch(e) { console.error(e); }
  // The expert page is the sole live client — we only flip the session to live
  // and hand off; we do NOT open a monitor WebSocket here (single-connection).
  window.open(`/expert?session=${sessionId}`, '_blank');
}
```

- [ ] **Step 4: JS syntax check**

Run:
```bash
sed -n '/<script>/,/<\/script>/p' static/index.html | sed '1d;$d' > /tmp/index.js && node --check /tmp/index.js && echo OK
```
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add expert_call_agent/static/index.html
git commit -m "static: add Start Expert Call button to the operator console

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: End-to-end smoke test

**Files:** none (verification only)

- [ ] **Step 1: Run the full expert flow over the tunnel**

Start the server (`python3 run.py`). Over the IAP tunnel:

1. Open `http://localhost:8899/` (operator console).
2. **Use Sample Brief** → wait for context → **Generate Call Guide** → wait for guide.
3. Click **Start Expert Call** → a new tab opens at `/expert?session=<id>`.
4. In the expert tab, click **Join call** and grant mic permission.
5. **Confirm:** after ~2 seconds, Shaun greets you — you hear TTS audio, the waveform reacts to the voice, the state shows "Speaking…", and an agent bubble appears in the transcript. State returns to "Listening…" when the audio ends.
6. Either speak an answer (mic — it auto-sends ~0.8 s after you stop talking, via the VAD endpointer) or type one in the fallback box + Enter. Confirm an expert (blue, right) bubble appears, state goes "Thinking…", and Shaun follows up with the next question (audio + bubble) — i.e. the agent drives.
7. Click **Leave call** → mic stops, state shows "Call ended".

Expected: all of the above; the only acceptable console noise is a favicon 404.

- [ ] **Step 2: Confirm no regression on the operator console demo path**

In the operator tab (fresh session: reload, Use Sample Brief, Generate Guide), click **Run Demo Simulation** and confirm the canned demo still streams transcript/ai_turn/notes/coverage as before. (Use one path at a time — don't run the demo while an expert tab is live on the same session.)

Expected: demo behaves exactly as before this feature.

- [ ] **Step 3: Final commit (if any verification fixes were needed)**

If Steps 1-2 surfaced fixes, commit them:

```bash
git add -A
git commit -m "fix: address expert-call smoke-test findings

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

If no fixes were needed, skip this step.

---

## Done

After Task 9, the feature is complete: a remote expert opens `/expert`, joins, and is interviewed by an agent that opens and drives the conversation — voice + waveform + live transcript — with the operator console used only for prep. Push to `main` when the user asks.
```
