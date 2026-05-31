import asyncio
from models import CallSession, TranscriptEntry, StructuredNote, CoverageStatus


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, CallSession] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._active_calls: set[str] = set()

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
            return session

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
