import asyncio
import base64
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, HTTPException

from agents.qa_agent import QAAgent
from agents.followup_agent import FollowUpAgent
from agents.note_taker import NoteTakerAgent
from agents.orchestrator import OrchestratorAgent
from live_orchestration import LiveCallQueue
from models import TranscriptEntry, AgentAction
import config

router = APIRouter()

# Frames the read-only monitor channel does not need: audio is heavy and useless
# to the operator panels, and demo progress is test-path noise.
_MONITOR_SKIP_TYPES = {"tts_audio", "demo_progress", "demo_complete"}

DEMO_EXPERT_RESPONSES = [
    "Thanks for having me. I was the VP of Infrastructure at CoreWeave for about two years before I moved on last October. I reported directly to the CTO, Brian Venturo, and was responsible for our data center buildout and GPU cluster operations.",
    "So in terms of total GPU fleet — when I left we had roughly 40,000 H100s deployed across six operational data centers, with another 25,000 on order for the new facilities in Dallas and Chicago. We also still had about 14,000 A100s running legacy workloads that customers hadn't migrated yet. The B200 rollout was just beginning — we had early access units in our Weehawken facility for internal testing.",
    "The business was scaling incredibly fast. Revenue run-rate was around 1.8 billion when I left, up from maybe 500 million a year earlier. But here's the thing — gross margins were thinner than people think. On paper you see 60 percent, but once you factor in power costs, networking equipment amortization, and the liquid cooling infrastructure, real gross margin on deployed capacity was closer to 42 to 45 percent.",
    "Customer concentration was our biggest strategic risk. Microsoft alone was roughly 35 percent of revenue through their Azure partnership. Then you had maybe three or four major AI labs — I can't name them specifically — but the top five customers combined were about 62 percent of total revenue. We were actively trying to diversify into enterprise but that sales cycle is much longer.",
    "NVIDIA allocation is everything in this market. Our relationship with Jensen was genuinely strong — CoreWeave was one of the first cloud providers to go all-in on NVIDIA before the AI boom. That earned us priority allocation. But it's not just about the relationship. NVIDIA looks at how many GPUs you're actually deploying, your utilization rates, and whether you're building the full DGX-compatible stack with InfiniBand networking. If you cut corners on networking, they notice.",
    "Power is becoming the real bottleneck, not GPUs. We were paying between 4.5 to 6 cents per kilowatt hour on our best long-term PPAs, but the newer facilities were coming in at 7 to 8 cents because power markets have tightened. A single H100 rack pulls about 40 kilowatts with liquid cooling. When you're deploying thousands of racks, your power bill is massive. Crusoe has a real advantage here with their stranded energy play — I've heard they're getting power below 3 cents.",
    "On the competitive landscape — Lambda is strong with the developer community but they're much smaller, maybe a quarter our scale. Together AI is interesting but they're more of an inference play. The real competition is Azure and GCP. Azure is tough because they have the OpenAI partnership and massive enterprise distribution. But their GPU clusters are often oversubscribed and their InfiniBand networking isn't as tight as ours. We consistently won on raw training performance.",
    "Interconnect matters enormously for training workloads. We ran full-fat InfiniBand across all our H100 clusters — 400 gig per node. Some competitors use RoCE Ethernet to save money, and it works fine for inference, but for distributed training across hundreds or thousands of GPUs, the latency difference is night and day. Customers doing large-scale training would choose us over Azure specifically for networking performance.",
    "The B200 Blackwell transition is going to reshuffle the market. The power draw per GPU is significantly higher — roughly 1,000 watts per chip versus 700 for H100. So every data center needs to be re-evaluated for power and cooling capacity. Companies that built facilities optimized for H100 density may need significant retrofits. This is where having newer facilities with flexible power and liquid cooling infrastructure gives you an advantage.",
    "Looking ahead, I think the market will consolidate around four or five major players. The hyperscalers — AWS, Azure, GCP — will always be there. CoreWeave has the scale and NVIDIA relationship to be the leading independent. Then you'll have one or two specialists surviving in niches — maybe Crusoe on the energy cost side, or Cerebras if their wafer-scale approach gains traction for specific workloads. The mid-tier players like Voltage Park and Applied Digital will either get acquired or struggle to compete on both price and performance.",
]


def _err(exc: Exception) -> str:
    msg = str(exc)
    return f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__


async def _run_agent(coro):
    """Run a live-call agent with a hard timeout so a hung LLM call can't stall the loop."""
    return await asyncio.wait_for(coro, timeout=config.AGENT_TIMEOUT_SECONDS)


async def speak(emit, app, session_id: str, utterance: str, *, agent_name: str,
                flag_type: str, rationale: str) -> None:
    """Emit one agent turn: ai_turn event, interviewer transcript entry, TTS audio.

    Shared by the reactive turn loop (process_entry) and the proactive opening so
    both produce an identical client-facing sequence. `emit` fans each message out
    to the expert socket and any read-only monitors.
    """
    sessions = app.state.sessions
    await emit({
        "type": "ai_turn",
        "question": utterance,
        "agent": agent_name,
        "flag_type": flag_type,
        "rationale": rationale,
    })

    ai_entry = TranscriptEntry(speaker="interviewer", text=utterance)
    await sessions.add_transcript_entry(session_id, ai_entry)
    await emit({"type": "transcript", "entry": ai_entry.model_dump()})

    try:
        audio_bytes = await app.state.tts.synthesize(utterance)
        await emit({
            "type": "tts_audio",
            "data": base64.b64encode(audio_bytes).decode(),
        })
    except Exception:
        pass


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


@router.post("/start/{session_id}")
async def start_call(request: Request, session_id: str):
    sessions = request.app.state.sessions
    session = await sessions.get_session(session_id)
    if not session.call_guide:
        raise HTTPException(400, "Generate a call guide first")
    await sessions.update_session(session_id, status="live")
    return {"status": "live", "session_id": session_id}


@router.post("/end/{session_id}")
async def end_call(request: Request, session_id: str):
    sessions = request.app.state.sessions
    await sessions.update_session(session_id, status="post_call")
    return {"status": "post_call", "session_id": session_id}


@router.get("/active")
async def active_call(request: Request):
    """The live session id for the expert page to auto-join (None if no live call)."""
    sessions = request.app.state.sessions
    return {"session_id": sessions.get_live_session_id()}


@router.websocket("/monitor/{session_id}")
async def monitor_websocket(ws: WebSocket, session_id: str):
    """Read-only live view of a call for the operator console.

    Subscribes to the same outbound stream the expert receives (minus audio and
    demo frames) without taking the single-call lock — so it never blocks the
    live expert and any number of monitors may attach. Replays current state on
    connect so a monitor joining mid-call (or after a refresh) sees the history.
    """
    await ws.accept()
    sessions = ws.app.state.sessions

    try:
        session = await sessions.get_session(session_id)
    except KeyError:
        await ws.send_json({"type": "error", "message": "Session not found"})
        await ws.close()
        return

    for entry in session.transcript:
        await ws.send_json({"type": "transcript", "entry": entry.model_dump()})
    for note in session.notes:
        await ws.send_json({"type": "note", "note": note.model_dump()})
    if session.coverage:
        await ws.send_json({
            "type": "coverage_update",
            "coverage": [c.model_dump() for c in session.coverage],
        })

    sessions.register_monitor(session_id, ws)
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            # Read-only: ignore any inbound payloads from the monitor.
    except WebSocketDisconnect:
        pass
    finally:
        sessions.unregister_monitor(session_id, ws)


@router.websocket("/ws/{session_id}")
async def call_websocket(ws: WebSocket, session_id: str):
    await ws.accept()

    app = ws.app
    sessions = app.state.sessions
    gemini = app.state.gemini
    claude = app.state.claude
    stt = app.state.stt

    try:
        session = await sessions.get_session(session_id)
    except KeyError:
        await ws.send_json({"type": "error", "message": "Session not found"})
        await ws.close()
        return

    # One live connection per session: a reconnect / second tab must not run a
    # parallel loop against shared session state.
    if not await sessions.acquire_call(session_id):
        await ws.send_json({"type": "error", "message": "Call already active for this session"})
        await ws.close()
        return

    async def emit(message: dict):
        """Send to the live expert, then fan out to any read-only monitors."""
        await ws.send_json(message)
        if message.get("type") not in _MONITOR_SKIP_TYPES:
            await sessions.broadcast_to_monitors(session_id, message)

    try:
        qa = QAAgent(gemini, claude)
        followup = FollowUpAgent(gemini, claude)
        note_taker = NoteTakerAgent(gemini, claude)
        orchestrator = OrchestratorAgent(gemini, claude)
        queue = LiveCallQueue()

        # On reconnect, don't re-extract notes for transcript entries already recorded.
        note_taker.seed_processed(len(session.transcript))

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
                    emit, app, session_id, opening,
                    agent_name="orchestrator",
                    flag_type="opening",
                    rationale="Opening the call",
                )

        async def process_entry(entry: TranscriptEntry):
            await sessions.add_transcript_entry(session_id, entry)
            session = await sessions.get_session(session_id)
            await emit({"type": "transcript", "entry": entry.model_dump()})

            qa_result, fu_result, nt_result = await asyncio.gather(
                _run_agent(qa.run(session)),
                _run_agent(followup.run(session)),
                _run_agent(note_taker.run(session)),
                return_exceptions=True,
            )

            # Note-taker (passive) — record notes, never queued.
            if isinstance(nt_result, Exception):
                await emit({"type": "error", "message": _err(nt_result)})
            else:
                for note in note_taker.get_latest_notes():
                    await sessions.add_note(session_id, note)
                    await emit({"type": "note", "note": note.model_dump()})

            # Coverage status from the Follow-up agent.
            if isinstance(fu_result, Exception):
                await emit({"type": "error", "message": _err(fu_result)})
            else:
                coverage = followup.get_latest_coverage()
                if coverage:
                    await sessions.update_coverage(session_id, coverage)
                    await emit({
                        "type": "coverage_update",
                        "coverage": [c.model_dump() for c in coverage],
                    })

            # Collect flags from QA + Follow-up.
            flags: list[AgentAction] = []
            if isinstance(qa_result, AgentAction):
                flags.append(qa_result)
            elif isinstance(qa_result, Exception):
                await emit({"type": "error", "message": _err(qa_result)})
            if isinstance(fu_result, list):
                flags.extend(fu_result)

            # Deterministic selection.
            queue.tick()
            queue.enqueue(flags)
            selected = queue.select_next()
            if selected is None:
                return  # nothing worth raising -> stay silent

            # Compose the spoken turn.
            try:
                utterance = await _run_agent(orchestrator.run(session, flag=selected))
            except Exception as e:
                await emit({"type": "error", "message": _err(e)})
                return
            if not utterance or not utterance.strip():
                return
            utterance = utterance.strip()

            await speak(
                emit, app, session_id, utterance,
                agent_name=selected.agent_name,
                flag_type=selected.flag_type,
                rationale=selected.metadata.get("rationale")
                or selected.metadata.get("reason", ""),
            )

        while True:
            raw = await ws.receive()
            if raw.get("type") == "websocket.disconnect":
                break

            if "bytes" in raw:
                audio = raw["bytes"]
                if len(audio) < 100:
                    continue
                try:
                    text = await stt.transcribe(audio, mime_type="audio/webm")
                    text = text.strip()
                    if text:
                        entry = TranscriptEntry(speaker="expert", text=text)
                        await process_entry(entry)
                except Exception as e:
                    await emit({"type": "error", "message": f"STT: {_err(e)}"})

            elif "text" in raw:
                data = json.loads(raw["text"])

                if data.get("type") == "text_input":
                    entry = TranscriptEntry(
                        speaker=data.get("speaker", "expert"),
                        text=data["text"],
                    )
                    await process_entry(entry)

                elif data.get("type") == "run_demo":
                    for i, expert_text in enumerate(DEMO_EXPERT_RESPONSES):
                        await emit({
                            "type": "demo_progress",
                            "current": i + 1,
                            "total": len(DEMO_EXPERT_RESPONSES),
                        })
                        entry = TranscriptEntry(speaker="expert", text=expert_text)
                        await process_entry(entry)
                        await asyncio.sleep(1)

                    await emit({"type": "demo_complete"})

    except WebSocketDisconnect:
        pass
    finally:
        await sessions.release_call(session_id)
