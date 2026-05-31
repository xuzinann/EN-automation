from agents.base_agent import BaseAgent
from models import CallSession, AgentAction, StructuredNote


class NoteTakerAgent(BaseAgent):
    name = "note_taker"
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

    async def run(self, session: CallSession, **kwargs) -> list[AgentAction]:
        new_entries = session.transcript[self._last_processed:]
        if not new_entries:
            return []

        self._last_processed = len(session.transcript)
        transcript = "\n".join(f"[{e.speaker}] {e.text}" for e in new_entries)

        prompt = (
            f"Extract structured notes from these new transcript entries:\n\n{transcript}"
        )
        response = await self._call_model(prompt)
        parsed = self._parse_json(response)

        actions = []
        notes_list = parsed if isinstance(parsed, list) else parsed.get("notes", [])
        for note_data in notes_list:
            note = StructuredNote.model_validate(note_data)
            actions.append(AgentAction(
                agent_name=self.name,
                action_type="note",
                content=note.content,
                priority=0.5,
                metadata={
                    "category": note.category,
                    "confidence": note.confidence,
                    "source_quote": note.source_quote,
                },
            ))

        self._latest_notes = [
            StructuredNote.model_validate(n)
            for n in notes_list
        ]
        return actions

    def get_latest_notes(self) -> list[StructuredNote]:
        return getattr(self, "_latest_notes", [])
