from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

from agents.post_call_summarizer import PostCallSummarizerAgent

router = APIRouter()


@router.post("/summarize/{session_id}")
async def summarize(request: Request, session_id: str):
    sessions = request.app.state.sessions
    gemini = request.app.state.gemini
    claude = request.app.state.claude

    session = await sessions.get_session(session_id)
    if not session.transcript:
        raise HTTPException(400, "No transcript to summarize")

    agent = PostCallSummarizerAgent(gemini, claude)
    result = await agent.run(session)
    await sessions.update_session(
        session_id,
        final_summary=result["markdown"],
        key_takeaways=result["takeaways"],
        status="post_call",
    )

    return {"summary": result["markdown"], "takeaways": result["takeaways"]}


@router.get("/export/{session_id}")
async def export_session(request: Request, session_id: str):
    sessions = request.app.state.sessions
    session = await sessions.get_session(session_id)
    return JSONResponse(content=session.model_dump(), media_type="application/json")


@router.get("/notes/{session_id}")
async def get_notes(request: Request, session_id: str):
    sessions = request.app.state.sessions
    session = await sessions.get_session(session_id)
    return {"notes": [n.model_dump() for n in session.notes]}
