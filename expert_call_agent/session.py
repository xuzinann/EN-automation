import asyncio
from models import CallSession, TranscriptEntry, StructuredNote, CoverageStatus


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, CallSession] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._active_calls: set[str] = set()
        self._live_session_id: str | None = None
        self._monitors: dict[str, set] = {}

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

    def register_monitor(self, session_id: str, ws) -> None:
        """Add a read-only monitor socket for a session (does not take the call lock)."""
        self._monitors.setdefault(session_id, set()).add(ws)

    def unregister_monitor(self, session_id: str, ws) -> None:
        conns = self._monitors.get(session_id)
        if conns is not None:
            conns.discard(ws)
            if not conns:
                self._monitors.pop(session_id, None)

    async def broadcast_to_monitors(self, session_id: str, message: dict) -> None:
        """Best-effort fan-out of one outbound message to every monitor socket.

        Iterates a snapshot so a socket dropping mid-broadcast can't corrupt the
        set; any socket that errors on send is unregistered.
        """
        for ws in list(self._monitors.get(session_id, ())):
            try:
                await ws.send_json(message)
            except Exception:
                self.unregister_monitor(session_id, ws)
