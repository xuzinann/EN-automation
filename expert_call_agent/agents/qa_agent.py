from agents.base_agent import BaseAgent
from models import CallSession, AgentAction
from observability import op
import config

_VALID_FLAGS = {"must_ask", "should_ask", "nice_to_have"}


class QAAgent(BaseAgent):
    name = "qa_agent"
    temperature = config.STRUCTURED_TEMPERATURE
    max_tokens = config.LIVE_AGENT_MAX_TOKENS
    system_prompt = (
        "You are the lead interviewer in a PE expert call. "
        "Given the interview guide and real-time transcript, suggest the best next "
        "question to ask. Adapt based on what the expert has said — don't repeat "
        "questions already answered, and probe deeper when answers are vague.\n\n"
        "If the expert's last answer is still unfolding and no new question is "
        "warranted yet, return an empty question string to stay silent.\n\n"
        "Return ONLY valid JSON:\n"
        '{"question": "the question to ask, or empty string to stay silent", '
        '"rationale": "why this question now", '
        '"guide_section": "which section this addresses", '
        '"priority": "must_ask|should_ask|nice_to_have"}\n'
        "Set priority from the guide question's own priority (default should_ask).\n"
        "No markdown fences or commentary."
    )

    @op()
    async def run(self, session: CallSession, **kwargs) -> AgentAction | None:
        guide_json = session.call_guide.model_dump_json() if session.call_guide else "{}"
        recent = session.transcript[-config.MAX_TRANSCRIPT_CONTEXT:]
        transcript = "\n".join(f"[{e.speaker}] {e.text}" for e in recent)

        prompt = (
            f"Interview guide:\n{guide_json}\n\n"
            f"Transcript so far:\n{transcript}\n\n"
            "Suggest the single best next question to ask the expert, or an empty "
            "question if it's better to stay silent for now."
        )
        response = await self._call_model(prompt)
        parsed = self._safe_parse_json(response, {})

        question = (parsed.get("question") or "").strip()
        if not question:
            return None

        flag_type = parsed.get("priority", "should_ask")
        if flag_type not in _VALID_FLAGS:
            flag_type = "should_ask"

        return AgentAction(
            agent_name=self.name,
            action_type="suggest_question",
            flag_type=flag_type,
            content=question,
            metadata={
                "rationale": parsed.get("rationale", ""),
                "section": parsed.get("guide_section", ""),
            },
        )
