from fastapi import APIRouter, UploadFile, File, Request, HTTPException

from agents.context_ingestion import ContextIngestionAgent
from agents.call_guide_drafter import CallGuideDrafterAgent

router = APIRouter()


@router.post("/upload-brief")
async def upload_brief(request: Request, file: UploadFile = File(...)):
    sessions = request.app.state.sessions
    gemini = request.app.state.gemini
    claude = request.app.state.claude

    content = await file.read()
    brief_text = content.decode("utf-8")

    session = await sessions.create_session()
    agent = ContextIngestionAgent(gemini, claude)
    context = await agent.run(session, brief_text=brief_text)
    await sessions.update_session(session.session_id, project_context=context)

    return {
        "session_id": session.session_id,
        "project_context": context.model_dump(),
    }


@router.post("/generate-guide/{session_id}")
async def generate_guide(request: Request, session_id: str):
    sessions = request.app.state.sessions
    gemini = request.app.state.gemini
    claude = request.app.state.claude

    session = await sessions.get_session(session_id)
    if not session.project_context:
        raise HTTPException(400, "Upload a brief first")

    agent = CallGuideDrafterAgent(gemini, claude)
    guide = await agent.run(session)
    await sessions.update_session(session_id, call_guide=guide)

    return {"call_guide": guide.model_dump()}


@router.get("/session/{session_id}")
async def get_session(request: Request, session_id: str):
    sessions = request.app.state.sessions
    session = await sessions.get_session(session_id)
    return session.model_dump()
