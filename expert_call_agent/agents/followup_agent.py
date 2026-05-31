from agents.base_agent import BaseAgent
from models import CallSession, AgentAction, CoverageStatus
import config


class FollowUpAgent(BaseAgent):
    name = "followup_agent"
    temperature = config.STRUCTURED_TEMPERATURE
    system_prompt = (
        "You monitor an expert call for coverage completeness. "
        "Track which interview guide sections have been addressed. "
        "Flag when the expert gives vague or incomplete answers needing follow-up. "
        "Detect contradictions with known data points.\n\n"
        "RAISE EACH CONCERN ONLY ONCE. The transcript you are given includes the "
        "INTERVIEWER's own earlier turns (speaker 'interviewer'). Before adding a "
        "follow-up or contradiction, check whether the interviewer has ALREADY "
        "raised that same concern earlier in the transcript. If it has, do NOT "
        "raise it again — even if the expert still hasn't answered it satisfactorily, "
        "assume it is noted and move on. Only return follow-ups and contradictions "
        "that are genuinely NEW.\n\n"
        "For each contradiction, also set a short, SPECIFIC `topic` slug naming the "
        "subject of the conflict (e.g. 'revenue', 'gross margin', 'gpu count', "
        "'prior employer'). Reuse the SAME topic wording if that same conflict ever "
        "recurs, and keep topics specific so different conflicts get different topics.\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "coverage": [{"section": "string", "covered_pct": 0.0-1.0, '
        '"answered_questions": ["strings"], "remaining_questions": ["strings"]}],\n'
        '  "followups": [{"question": "string", "reason": "string"}],\n'
        '  "contradictions": [{"topic": "string", "claim": "string", "conflicts_with": "string"}]\n'
        "}\n"
        "No markdown fences or commentary."
    )

    async def run(self, session: CallSession, **kwargs) -> list[AgentAction]:
        guide_json = session.call_guide.model_dump_json() if session.call_guide else "{}"
        context_json = session.project_context.model_dump_json() if session.project_context else "{}"
        recent = session.transcript[-config.MAX_TRANSCRIPT_CONTEXT:]
        transcript = "\n".join(f"[{e.speaker}] {e.text}" for e in recent)

        prompt = (
            f"Interview guide:\n{guide_json}\n\n"
            f"Known data points:\n{context_json}\n\n"
            f"Transcript so far:\n{transcript}\n\n"
            "Analyze coverage completeness. Suggest follow-ups and flag contradictions, "
            "but ONLY ones not already raised by the interviewer earlier in the transcript above."
        )
        response = await self._call_model(prompt)
        parsed = self._safe_parse_json(response, {})

        actions = []
        for fu in parsed.get("followups", []):
            question = (fu.get("question") or "").strip()
            if not question:
                continue
            actions.append(AgentAction(
                agent_name=self.name,
                action_type="followup",
                flag_type="probe",
                content=question,
                metadata={"reason": fu.get("reason", "")},
            ))

        for contradiction in parsed.get("contradictions", []):
            actions.append(AgentAction(
                agent_name=self.name,
                action_type="contradiction",
                flag_type="contradiction",
                content=(
                    f"There may be an inconsistency: "
                    f"{contradiction.get('claim', '')} vs "
                    f"{contradiction.get('conflicts_with', '')}. Could you clarify?"
                ),
                metadata={
                    "reason": "contradiction with known data",
                    "topic": contradiction.get("topic", ""),
                    "claim": contradiction.get("claim", ""),
                    "conflicts_with": contradiction.get("conflicts_with", ""),
                },
            ))

        self._latest_coverage = [
            CoverageStatus.model_validate(c)
            for c in parsed.get("coverage", [])
        ]

        return actions

    def get_latest_coverage(self) -> list[CoverageStatus]:
        return getattr(self, "_latest_coverage", [])
