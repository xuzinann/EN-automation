import asyncio
import base64
import json
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, HTTPException

from agents.qa_agent import QAAgent
from agents.followup_agent import FollowUpAgent
from agents.note_taker import NoteTakerAgent
from agents.orchestrator import OrchestratorAgent
from live_orchestration import LiveCallQueue
from models import TranscriptEntry, AgentAction

router = APIRouter()

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

    qa = QAAgent(gemini, claude)
    followup = FollowUpAgent(gemini, claude)
    note_taker = NoteTakerAgent(gemini, claude)
    orchestrator = OrchestratorAgent(gemini, claude)
    queue = LiveCallQueue()

    async def process_entry(entry: TranscriptEntry):
        await sessions.add_transcript_entry(session_id, entry)
        session = await sessions.get_session(session_id)
        await ws.send_json({"type": "transcript", "entry": entry.model_dump()})

        qa_result, fu_result, nt_result = await asyncio.gather(
            qa.run(session),
            followup.run(session),
            note_taker.run(session),
            return_exceptions=True,
        )

        # Note-taker (passive) — record notes, never queued.
        if isinstance(nt_result, Exception):
            await ws.send_json({"type": "error", "message": str(nt_result)})
        else:
            for note in note_taker.get_latest_notes():
                await sessions.add_note(session_id, note)
                await ws.send_json({"type": "note", "note": note.model_dump()})

        # Coverage status from the Follow-up agent.
        if not isinstance(fu_result, Exception):
            coverage = followup.get_latest_coverage()
            if coverage:
                await sessions.update_coverage(session_id, coverage)
                await ws.send_json({
                    "type": "coverage_update",
                    "coverage": [c.model_dump() for c in coverage],
                })

        # Collect flags from QA + Follow-up.
        flags: list[AgentAction] = []
        if isinstance(qa_result, AgentAction):
            flags.append(qa_result)
        elif isinstance(qa_result, Exception):
            await ws.send_json({"type": "error", "message": str(qa_result)})
        if isinstance(fu_result, list):
            flags.extend(fu_result)
        elif isinstance(fu_result, Exception):
            await ws.send_json({"type": "error", "message": str(fu_result)})

        # Deterministic selection.
        queue.enqueue(flags)
        selected = queue.select_next()
        if selected is None:
            return  # empty queue -> stay silent, let the expert continue

        # Compose the spoken turn.
        session = await sessions.get_session(session_id)
        utterance = await orchestrator.run(session, flag=selected)
        if not utterance or not utterance.strip():
            return
        utterance = utterance.strip()

        await ws.send_json({
            "type": "ai_turn",
            "question": utterance,
            "agent": selected.agent_name,
            "flag_type": selected.flag_type,
            "rationale": selected.metadata.get("rationale") or selected.metadata.get("reason", ""),
        })

        ai_entry = TranscriptEntry(speaker="interviewer", text=utterance)
        await sessions.add_transcript_entry(session_id, ai_entry)
        await ws.send_json({"type": "transcript", "entry": ai_entry.model_dump()})

        # Auto-speak via TTS.
        try:
            audio_bytes = await app.state.tts.synthesize(utterance)
            await ws.send_json({
                "type": "tts_audio",
                "data": base64.b64encode(audio_bytes).decode(),
            })
        except Exception:
            pass

    try:
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
                    await ws.send_json({"type": "error", "message": f"STT: {e}"})

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
                        await ws.send_json({
                            "type": "demo_progress",
                            "current": i + 1,
                            "total": len(DEMO_EXPERT_RESPONSES),
                        })
                        entry = TranscriptEntry(speaker="expert", text=expert_text)
                        await process_entry(entry)
                        await asyncio.sleep(1)

                    await ws.send_json({"type": "demo_complete"})

    except WebSocketDisconnect:
        pass
