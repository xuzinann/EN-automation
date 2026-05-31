import asyncio
from models import CallSession, TranscriptEntry, StructuredNote, CoverageStatus


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, CallSession] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._active_calls: set[str] = set()
        self._live_session_id: str | None = None

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]

    async def create_session(self) -> CallSession:
        session = CallSession()
        self._sessions[session.session_id] = session
        return session

    async def get_session(self, session_id: str) -> CallSession:
        if session_id not in self._sessions:
            raise KeyError(f"Session {session_id} not found")
        return self._sessions[session_id]

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

    async def add_transcript_entry(self, session_id: str, entry: TranscriptEntry):
        async with self._get_lock(session_id):
            self._sessions[session_id].transcript.append(entry)

    async def add_note(self, session_id: str, note: StructuredNote):
        async with self._get_lock(session_id):
            self._sessions[session_id].notes.append(note)

    async def update_coverage(self, session_id: str, coverage: list[CoverageStatus]):
        async with self._get_lock(session_id):
            self._sessions[session_id].coverage = coverage

    async def acquire_call(self, session_id: str) -> bool:
        """Claim the single live-call slot for a session. False if already active."""
        async with self._get_lock(session_id):
            if session_id in self._active_calls:
                return False
            self._active_calls.add(session_id)
            return True

    async def release_call(self, session_id: str) -> None:
        async with self._get_lock(session_id):
            self._active_calls.discard(session_id)
