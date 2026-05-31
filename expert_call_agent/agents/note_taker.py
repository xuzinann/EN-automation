from agents.base_agent import BaseAgent
from models import CallSession, AgentAction, StructuredNote
import config


class NoteTakerAgent(BaseAgent):
    name = "note_taker"
    temperature = config.STRUCTURED_TEMPERATURE
    max_tokens = config.LIVE_AGENT_MAX_TOKENS
    system_prompt = (
        "Extract structured notes from an expert call transcript. "
        "Categorize each note:\n"
        "- headcount: N-sizing data by role\n"
        "- org_structure: reporting lines, departments\n"
        "- market: competitive positioning, market share\n"
        "- operations: processes, technology, efficiency\n"
        "- financials: revenue, costs, margins\n"
        "- quote: verbatim quote worth preserving\n"
        "- other: anything else noteworthy\n\n"
        "Assess confidence: stated (expert said directly), estimated (implied), "
        "inferred (you derived).\n\n"
        "Return ONLY valid JSON array:\n"
        '[{"category": "string", "content": "string", "confidence": "stated|estimated|inferred", '
        '"source_quote": "string or null"}]\n'
        "No markdown fences or commentary. Return [] if no new notes."
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_processed = 0
        self._latest_notes: list[StructuredNote] = []

    def seed_processed(self, count: int) -> None:
        """Skip note extraction for entries already in the transcript (reconnect)."""
        self._last_processed = count

    async def run(self, session: CallSession, **kwargs) -> list[AgentAction]:
        processed_through = len(session.transcript)
        new_entries = session.transcript[self._last_processed:processed_through]

        # Only extract facts from the expert — never from the AI's own questions.
        expert_entries = [e for e in new_entries if e.speaker == "expert"]
        if not expert_entries:
            # Nothing to extract (e.g. interviewer-only turns); skip them.
            self._last_processed = processed_through
            self._latest_notes = []
            return []

        transcript = "\n".join(f"[{e.speaker}] {e.text}" for e in expert_entries)

        prompt = (
            f"Extract structured notes from these new transcript entries:\n\n{transcript}"
        )
        response = await self._call_model(prompt)
        # Advance the watermark only after a successful call, so a timed-out or
        # failed extraction retries these entries next turn instead of dropping
        # their notes silently.
        self._last_processed = processed_through
        parsed = self._safe_parse_json(response, [])

        notes_list = parsed if isinstance(parsed, list) else parsed.get("notes", [])

        actions = []
        latest = []
        for note_data in notes_list:
            try:
                note = StructuredNote.model_validate(note_data)
            except Exception:
                continue
            latest.append(note)
            actions.append(AgentAction(
                agent_name=self.name,
                action_type="note",
                content=note.content,
                metadata={
                    "category": note.category,
                    "confidence": note.confidence,
                    "source_quote": note.source_quote,
                },
            ))

        self._latest_notes = latest
        return actions

    def get_latest_notes(self) -> list[StructuredNote]:
        return self._latest_notes
