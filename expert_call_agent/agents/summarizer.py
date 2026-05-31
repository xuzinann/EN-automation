import json
from agents.base_agent import BaseAgent
from models import CallSession, AgentAction
import config


class KeyTakeawaySummarizerAgent(BaseAgent):
    name = "summarizer"
    system_prompt = (
        "You synthesize key takeaways from an ongoing expert call. Focus on:\n"
        "1. Top findings that are new or surprising\n"
        "2. Data points that confirm or contradict the PE client's hypotheses\n"
        "3. Remaining gaps that still need answers\n\n"
        "Return ONLY valid JSON:\n"
        '{"takeaways": ["string"], "surprises": ["string"], "remaining_gaps": ["string"]}\n'
        "No markdown fences or commentary."
    )

    async def run(self, session: CallSession, **kwargs) -> AgentAction:
        notes_json = json.dumps([n.model_dump() for n in session.notes[-20:]])
        recent = session.transcript[-config.MAX_TRANSCRIPT_CONTEXT:]
        transcript = "\n".join(f"[{e.speaker}] {e.text}" for e in recent)

        hypotheses = ""
        if session.project_context:
            hypotheses = "\n".join(session.project_context.key_hypotheses)

        prompt = (
            f"Client hypotheses:\n{hypotheses}\n\n"
            f"Notes extracted so far:\n{notes_json}\n\n"
            f"Recent transcript:\n{transcript}\n\n"
            "Synthesize the key takeaways from this expert call."
        )
        response = await self._call_model(prompt)
        parsed = self._parse_json(response)

        self._latest_takeaways = parsed.get("takeaways", [])

        return AgentAction(
            agent_name=self.name,
            action_type="summary_update",
            content=json.dumps(parsed),
            priority=0.4,
            metadata={
                "surprises": parsed.get("surprises", []),
                "remaining_gaps": parsed.get("remaining_gaps", []),
            },
        )

    def get_latest_takeaways(self) -> list[str]:
        return getattr(self, "_latest_takeaways", [])
