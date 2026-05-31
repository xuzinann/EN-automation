import sys
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from clients.gemini_client import GeminiClient
from clients.claude_client import ClaudeClient
from clients.stt_client import STTClient
from clients.cloud_stt_client import CloudSTTClient
from clients.tts_client import TTSClient
from session import SessionManager
from api.routes_precall import router as precall_router
from api.routes_call import router as call_router
from api.routes_postcall import router as postcall_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.gemini = GeminiClient()
    app.state.claude = ClaudeClient()
    app.state.stt = (
        CloudSTTClient() if config.STT_PROVIDER == "chirp" else STTClient(app.state.gemini)
    )
    app.state.tts = TTSClient()
    app.state.sessions = SessionManager()
    yield
    await app.state.claude.close()
    await app.state.tts.close()
    await app.state.stt.close()


def create_app() -> FastAPI:
    app = FastAPI(title="Expert Call Virtual Agent", lifespan=lifespan)

    app.include_router(precall_router, prefix="/api/precall")
    app.include_router(call_router, prefix="/api/call")
    app.include_router(postcall_router, prefix="/api/postcall")

    base_dir = Path(__file__).resolve().parent.parent
    app.mount("/sample_data", StaticFiles(directory=str(base_dir / "sample_data")), name="sample_data")
    app.mount("/context", StaticFiles(directory=str(base_dir / "context")), name="context")

    async def expert_page():
        return FileResponse(str(base_dir / "static" / "expert.html"))

    app.add_api_route("/expert", expert_page, methods=["GET"], include_in_schema=False)

    app.mount("/", StaticFiles(directory=str(base_dir / "static"), html=True), name="static")

    return app
