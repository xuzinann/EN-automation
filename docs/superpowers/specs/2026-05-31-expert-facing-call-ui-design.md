# Expert-Facing Live Call UI — Design Spec

**Date:** 2026-05-31
**Status:** Approved design, ready for implementation plan
**Author:** brainstormed with the team (visual companion session)

## Goal

Add a second, self-contained web page — `static/expert.html`, served at `/expert` — that lets a **remote expert** take the live call directly with the AI interviewer ("Shaun"). The expert speaks into their mic and hears the agent's voice, visualized by a waveform. The agent **drives** the conversation: it opens the call proactively and leads through the guide. No human operator is in the loop during the call.

## Architecture (one sentence)

The expert page is the **sole live WebSocket client**; it reuses the existing per-connection call loop in `routes_call.py` (STT on the inbound mic stream, agents → queue → orchestrator → TTS on the way out), with two backend additions — a **proactive opening** (the agent speaks first when the expert joins) and a **`GET /api/call/active`** lookup so the page can auto-join the live session.

## Tech Stack

- Backend: FastAPI + WebSockets (existing), Vertex AI via ADC (Gemini, Claude, Cloud STT/TTS) — unchanged.
- Frontend: single static HTML page, vanilla JS + WebAudio API (no framework, no build), matching `index.html`'s style.

---

## Scope & Decisions

This is a **hackathon demo**. Explicit decisions made during brainstorming:

| Decision | Choice |
|---|---|
| Expert audio | **Full two-way participant** — mic → STT, agent TTS → playback + viz |
| Who's connected during the call | **Only the expert** (no operator monitoring, no human-in-the-loop) |
| Fan-out / multi-client | **Dropped** — single live connection; reuse existing per-connection loop |
| Session join | **Auto-join the active live session** via `GET /api/call/active`; honor `?session=<id>` as an override |
| Auth | **None** — single reachable URL |
| Voice visualizer | **Waveform** (scrolling oscilloscope line through a central agent orb) |
| Layout | **Split pane** — dark "stage" on top, **light** transcript reading-pane below |
| Transcript content | **Interleaved** — agent (Shaun) + expert's own transcribed answers |
| Agent behavior | **Agent opens after a ~2 s settle delay** (greets + asks first question via TTS before the expert speaks), then leads |

### Non-goals (out of scope)

- Operator monitoring / multi-client broadcast (deliberately dropped; the per-connection loop already handles a single client).
- Authentication, multi-session routing, or a join/lobby screen.
- Persisting pre-call edits, changing the post-call summary flow, or touching pre-call ingestion.
- Replacing the existing `index.html` test path (Run Demo, manual Start Call) — left intact; the new flow is additive.

---

## Data Flow

```
 Expert mic ─ webm/opus bytes ─► [WS receive loop]  (routes_call.py, ONE client)
                                      │  STT.transcribe(bytes) → text     ← STT: connection edge
                                      ▼
                                 TranscriptEntry(speaker="expert")
                                      │
                                      ▼  process_entry()
                          QA + Follow-up + Note-taker (asyncio.gather)
                                      │  flags → LiveCallQueue → select_next()
                                      ▼  Orchestrator composes ONE utterance
                                 speak(utterance):                          ← TTS: turn composition
                                   • send transcript (speaker="interviewer")
                                   • send ai_turn
                                   • TTS.synthesize(utterance) → send tts_audio
                                      │
                                      ▼ (over the same WS)
                                 Expert page renders + plays + visualizes
```

**Proactive opening** (new): when the expert's WS connects to a `live` session that has **no interviewer turn yet**, the agent waits a short settle delay (`config.LIVE_OPENING_DELAY_SECONDS`, default **2 s** — lets the page render, the audio context unlock, and the mic spin up) and then runs `speak(opening_utterance)` before entering the receive loop, where `opening_utterance = guide.opening_script + first guide question`. This is the only proactive turn; all subsequent turns are driven by the existing expert-answer → QA/queue/orchestrator loop, which already proposes the next guide question each turn.

---

## Components / Files

### Create

- **`static/expert.html`** — the expert page (UI + WS client + mic + waveform). Self-contained, like `index.html`.

### Modify

- **`api/routes_call.py`**
  - Extract the "speak one turn" tail of `process_entry` into a reusable helper `speak(ws, app, session_id, utterance, agent_name, flag_type, rationale)` (sends `ai_turn`, appends + sends the `interviewer` transcript entry, synthesizes + sends `tts_audio`). `process_entry` calls it; the opening calls it.
  - In `call_websocket`, after setup and before the receive loop: if `session.status == "live"`, a `call_guide` exists, and the transcript has **no** `interviewer` entry yet, `await asyncio.sleep(config.LIVE_OPENING_DELAY_SECONDS)` (default 2 s), then compose and `speak()` the opening (`opening_script` + first available question from `opening_questions` → else first deep-dive question → else QA's suggestion).
  - Add `GET /api/call/active` → `{"session_id": <live id or null>}`.
- **`session.py` (`SessionManager`)**
  - Track the live session: set `_live_session_id` when a session transitions to `status == "live"`; clear it when it leaves `live`. Add `get_live_session_id()`.
- **`api/app.py`**
  - Add an explicit `GET /expert` route returning `FileResponse(static/expert.html)`, registered **before** the catch-all `/` static mount so it resolves cleanly.
- **`static/index.html`**
  - Add a **"Start Expert Call"** button (enabled after the guide is generated): `POST /api/call/start/{sessionId}` then reveal/open `/expert?session=<id>`. It does **not** open a WebSocket (so it won't collide with the expert's single connection). Existing Start Call / Run Demo / mic controls remain for the headless test path.

---

## WebSocket Contract (reused — no changes)

Server → client (the expert page consumes a subset):
- `transcript` `{entry:{speaker, text, timestamp}}` — **rendered as bubbles** (interviewer left/white "Shaun", expert right/blue "You").
- `ai_turn` `{question, agent, flag_type, rationale}` — used only to set state to **Speaking…** and emphasize the orb (content already arrives via `transcript`, so it is **not** rendered again — avoids double bubbles).
- `tts_audio` `{data: base64 mp3}` — played and routed through a WebAudio `AnalyserNode` to drive the waveform amplitude.
- `note`, `coverage_update` — **ignored** on the expert page (operator-facing).
- `error` — logged; optional subtle inline notice.
- `demo_progress` / `demo_complete` — ignored (test path only).

Client → server:
- raw audio **bytes** (mic chunks, webm/opus, every 5 s) → STT.
- (`text_input`, `run_demo` exist but the expert page does not send them.)

---

## Frontend Spec — `expert.html`

**Join flow:** on load, read `?session=`; if absent, poll `GET /api/call/active` (~1 s interval) until a `session_id` is returned, showing a **"Waiting for the call to start…"** state. Then open the WS to `/api/call/ws/{session_id}`.

**Layout** (matches approved mockup `layout-v2`):
- **Top stage (dark, ~54% height):** centered agent **orb** (gradient circle) with a **scrolling waveform** behind it, a `● LIVE` indicator, the agent name ("Shaun"), and a **state line**. Radial dark gradient background.
- **Bottom transcript (light reading pane):** header "Transcript"; interleaved bubbles — Shaun's questions (white, left-aligned, green label) and the expert's transcribed answers (light-blue, right-aligned, blue label). Auto-scrolls to newest.
- **Footer (light):** mic status ("Mic on — the agent hears you") + a **Leave call** button.

**State machine** (drives the state line + waveform):
- **Listening…** — mic active, agent idle (default between turns).
- **Thinking…** — set when an `expert` transcript entry appears (agent is processing).
- **Speaking…** — set on `ai_turn` / while `tts_audio` is playing.
- On TTS `ended` → back to **Listening…**.

**Waveform behavior:** idle = calm low-amplitude animation; while TTS is playing, amplitude is driven by an `AnalyserNode` tap on the playback node, so it genuinely visualizes the agent's voice. (The waveform represents the **agent's** voice, per the brief — the expert's speaking is reflected by the "Listening…" state, not the waveform.)

**Mic:** reuse `index.html`'s capture pattern — `getUserMedia` → `MediaRecorder` (`audio/webm;codecs=opus`, 5 s chunks) → `ws.send(blob)`. Auto-start on connect; if unavailable, show a text-input fallback (as `index.html` does).

**Leave call:** stop mic, close WS, show a simple "Call ended." end state.

---

## Backend Spec — details

- **`GET /api/call/active`** (in `routes_call.py`, prefix `/api/call`): returns `{"session_id": sessions.get_live_session_id()}` (value may be `null`). No body required.
- **Live-session tracking** (`SessionManager`): a single `_live_session_id` is sufficient given the demo's "one live call at a time" assumption. Set on transition to `live` (in `start_call` / `update_session`), cleared on transition away from `live` (in `end_call`). `get_live_session_id()` returns it only if that session still has `status == "live"`.
- **Opening delay:** new config knob `LIVE_OPENING_DELAY_SECONDS = 2.0` (in `config.py`), awaited before the opening turn. Tunable; set to `0` to disable.
- **Opening idempotency:** "no interviewer turn in `session.transcript`" is the guard — it survives reconnects (the agent won't re-open) and needs no extra state.
- **Driving:** no new progression engine. After the opening, the existing `process_entry` path runs every expert turn and the QA agent proposes the next guide question, so the agent keeps leading. If QA stays silent **and** the queue is empty, the agent waits for the expert (acceptable for the demo).

---

## Error Handling

- **No active call yet:** expert page shows "Waiting for the call to start…" and keeps polling `/active`.
- **WS drops:** `onclose` stops the mic and shows a "Disconnected" notice; the existing opening-idempotency guard means a manual refresh/reconnect resumes without re-greeting.
- **TTS failure:** already swallowed server-side (`process_entry` wraps TTS in try/except); the transcript bubble still appears, so the expert can read the question — which is the whole point of the reading pane.
- **STT failure:** surfaced as an `error` message (existing behavior); the loop continues.
- **Agent timeouts:** unchanged — the hardening layer's `asyncio.wait_for` per agent still applies.

---

## Testing (hackathon-light)

Per project convention, skip heavy automated tests. Verify with:
1. `node --check` on the inline JS of `expert.html` (syntax).
2. A tiny check that `GET /api/call/active` returns `null` before a call and the live id after `POST /start` (in-process `TestClient`).
3. Manual smoke over the IAP tunnel: prep on `index.html` → "Start Expert Call" → open `/expert` → confirm the agent greets immediately (audio + waveform + transcript bubble), then answer (mic or text) and confirm the agent drives with follow-ups.

---

## Open Questions

None — all design decisions resolved during brainstorming.
