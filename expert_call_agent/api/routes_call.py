import asyncio
import base64
import json
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, HTTPException

from agents.qa_agent import QAAgent
from agents.followup_agent import FollowUpAgent
from agents.note_taker import NoteTakerAgent
from agents.summarizer import KeyTakeawaySummarizerAgent
from agents.orchestrator import OrchestratorAgent
from models import TranscriptEntry

router = APIRouter()

DEMO_EXPERT_RESPONSES = [
    "Thanks for having me. I was the Director of Cloud Operations at TechServ for about three years before I left last September. I reported directly to Sarah Chen.",
    "In terms of headcount, when I left we had roughly 620 people total. But that's a bit misleading because about 80 to 90 of those were contractors, mainly in the help desk and basic infrastructure monitoring roles. So full-time employees were closer to 530 to 540.",
    "Engineering was the largest group. We had about 180 full-time engineers broken into three main teams: cloud infrastructure which was about 70 people, the ServeWatch platform team which was around 45, and then cybersecurity consulting which had about 65 engineers. The rest were spread across DevOps and internal tools.",
    "The sales team was actually quite lean, maybe 40 to 45 people including account managers. Most of our growth came from existing client expansion rather than new logo acquisition. We had a dedicated client success team of about 15 people who drove upsells.",
    "Org structure below the C-suite — so you had James Morrison as CTO, Sarah Chen as VP Engineering reporting to him. Under Sarah there were three Senior Directors for each of the main engineering pillars. Then each Senior Director had two to three team leads. So it was basically four layers: CTO, VP, Senior Director, Team Lead, then individual contributors.",
    "Key person risk is real with James. He built the ServeWatch platform from scratch and a lot of the architectural decisions still flow through him. Sarah is operationally excellent but James is the technical visionary. If James left, I think you'd see some attrition in the senior engineering ranks within six months.",
    "Attrition was actually better than industry average. We were running about 14 to 15 percent annually for engineering, which is quite good for IT services. Help desk was higher, around 22 percent, but that's typical. The main reason people stayed was the culture and the fact that the work was genuinely interesting.",
    "ServeWatch — look, I'll be honest. The core monitoring engine uses Prometheus and Grafana under the hood. But the value is in the custom alerting logic, the client-specific dashboards, and the automated remediation workflows that James and the team built on top. Those are genuinely proprietary and clients love them. It would take a competitor 18 to 24 months to replicate that layer.",
    "Client concentration is a concern. Regional National Bank alone was probably 18 to 20 percent of revenue. Add First Midwest Health System and Chicago Manufacturing Group and you're at about 45 percent from the top three. But all three have been clients for seven plus years with auto-renewing contracts.",
    "On the integration front, the biggest challenge would be the ServeWatch platform. Every client is deeply customized on it. You can't just swap in another monitoring stack without a major migration effort for each client. The rest of the business — sales, operations, help desk — would integrate fairly straightforwardly with another IT services firm.",
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
    summarizer = KeyTakeawaySummarizerAgent(gemini, claude)
    orchestrator = OrchestratorAgent(gemini, claude)

    async def process_entry(entry: TranscriptEntry):
        await sessions.add_transcript_entry(session_id, entry)
        session = await sessions.get_session(session_id)

        await ws.send_json({"type": "transcript", "entry": entry.model_dump()})

        results = await asyncio.gather(
            qa.run(session),
            followup.run(session),
            note_taker.run(session),
            summarizer.run(session),
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, Exception):
                await ws.send_json({"type": "error", "message": str(result)})
                continue
            actions = result if isinstance(result, list) else [result]
            for action in actions:
                await sessions.add_action(session_id, action)

        for note in note_taker.get_latest_notes():
            await sessions.add_note(session_id, note)
            await ws.send_json({"type": "note", "note": note.model_dump()})

        coverage = followup.get_latest_coverage()
        if coverage:
            await sessions.update_coverage(session_id, coverage)
            await ws.send_json({
                "type": "coverage_update",
                "coverage": [c.model_dump() for c in coverage],
            })

        takeaways = summarizer.get_latest_takeaways()
        if takeaways:
            await sessions.update_takeaways(session_id, takeaways)
            await ws.send_json({"type": "key_takeaways", "takeaways": takeaways})

        session = await sessions.get_session(session_id)
        orch_result = await orchestrator.run(session)
        await sessions.pop_pending_actions(session_id)

        selected = orch_result.get("selected_action")
        if selected:
            await ws.send_json({
                "type": "suggested_question",
                "question": selected.get("content", ""),
                "rationale": orch_result.get("reason", ""),
                "agent": selected.get("agent_name", ""),
                "action_type": selected.get("action_type", ""),
            })

    try:
        while True:
            raw = await ws.receive()
            if raw.get("type") == "websocket.disconnect":
                break

            if "bytes" in raw:
                audio = raw["bytes"]
                text = await stt.transcribe(audio)
                entry = TranscriptEntry(speaker="expert", text=text)
                await process_entry(entry)

            elif "text" in raw:
                data = json.loads(raw["text"])

                if data.get("type") == "text_input":
                    entry = TranscriptEntry(
                        speaker=data.get("speaker", "expert"),
                        text=data["text"],
                    )
                    await process_entry(entry)

                elif data.get("type") == "ask_question":
                    entry = TranscriptEntry(speaker="interviewer", text=data["question"])
                    await sessions.add_transcript_entry(session_id, entry)
                    await ws.send_json({"type": "transcript", "entry": entry.model_dump()})

                    tts = app.state.tts
                    try:
                        audio_bytes = await tts.synthesize(data["question"])
                        audio_b64 = base64.b64encode(audio_bytes).decode()
                        await ws.send_json({"type": "tts_audio", "data": audio_b64})
                    except Exception:
                        pass

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
