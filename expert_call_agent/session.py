import asyncio
from models import CallSession, TranscriptEntry, StructuredNote, AgentAction, CoverageStatus


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, CallSession] = {}
        self._locks: dict[str, asyncio.Lock] = {}

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

    async def add_action(self, session_id: str, action: AgentAction):
        async with self._get_lock(session_id):
            self._sessions[session_id].pending_actions.append(action)

    async def pop_pending_actions(self, session_id: str) -> list[AgentAction]:
        async with self._get_lock(session_id):
            actions = list(self._sessions[session_id].pending_actions)
            self._sessions[session_id].pending_actions.clear()
            return actions

    async def update_coverage(self, session_id: str, coverage: list[CoverageStatus]):
        async with self._get_lock(session_id):
            self._sessions[session_id].coverage = coverage

    async def update_takeaways(self, session_id: str, takeaways: list[str]):
        async with self._get_lock(session_id):
            self._sessions[session_id].key_takeaways = takeaways
