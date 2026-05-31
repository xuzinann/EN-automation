import json
from agents.base_agent import BaseAgent
from models import CallSession, AgentAction, CoverageStatus
import config


class FollowUpAgent(BaseAgent):
    name = "followup_agent"
    system_prompt = (
        "You monitor an expert call for coverage completeness. "
        "Track which interview guide sections have been addressed. "
        "Flag when the expert gives vague or incomplete answers needing follow-up. "
        "Detect contradictions with known data points.\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "coverage": [{"section": "string", "covered_pct": 0.0-1.0, '
        '"answered_questions": ["strings"], "remaining_questions": ["strings"]}],\n'
        '  "followups": [{"question": "string", "reason": "string", "priority": 0.0-1.0}],\n'
        '  "contradictions": [{"claim": "string", "conflicts_with": "string"}]\n'
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
            "Analyze coverage completeness, suggest follow-ups, and flag contradictions."
        )
        response = await self._call_model(prompt)
        parsed = self._parse_json(response)

        actions = []
        for fu in parsed.get("followups", []):
            actions.append(AgentAction(
                agent_name=self.name,
                action_type="followup",
                content=fu.get("question", ""),
                priority=fu.get("priority", 0.6),
                metadata={"reason": fu.get("reason", "")},
            ))

        for contradiction in parsed.get("contradictions", []):
            actions.append(AgentAction(
                agent_name=self.name,
                action_type="contradiction",
                content=f"Contradiction: {contradiction.get('claim', '')} vs {contradiction.get('conflicts_with', '')}",
                priority=0.9,
            ))

        self._latest_coverage = [
            CoverageStatus.model_validate(c)
            for c in parsed.get("coverage", [])
        ]

        return actions

    def get_latest_coverage(self) -> list[CoverageStatus]:
        return getattr(self, "_latest_coverage", [])
