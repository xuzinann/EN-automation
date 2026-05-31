from agents.base_agent import BaseAgent
from models import CallSession, AgentAction
import config

_INTENT = {
    "contradiction": "Tactfully surface this apparent inconsistency and ask the expert to clarify",
    "probe": "Probe deeper on the expert's most recent answer",
    "must_ask": "Ask this priority question",
    "should_ask": "Ask this question",
    "nice_to_have": "Ask this question only if it fits naturally",
}


class OrchestratorAgent(BaseAgent):
    name = "orchestrator"
    system_prompt = (
        "You are the spoken voice of an AI interviewer on a live PE expert call. "
        "You receive the single next point to raise — already chosen for you — plus the recent "
        "transcript. Phrase it as ONE natural, concise spoken turn: a brief acknowledgment or "
        "transition if it helps the flow, then the question. Do not add commentary, multiple "
        "questions, lists, or markdown. Return only the words to speak."
    )

    async def run(self, session: CallSession, flag: AgentAction | None = None, **kwargs) -> str:
        if flag is None:
            return ""

        recent = session.transcript[-config.MAX_TRANSCRIPT_CONTEXT:]
        transcript = "\n".join(f"[{e.speaker}] {e.text}" for e in recent)
        intent = _INTENT.get(flag.flag_type, "Ask this question")

        prompt = (
            f"Recent transcript:\n{transcript}\n\n"
            f"Next point to raise ({flag.flag_type}): {flag.content}\n\n"
            f"Instruction: {intent}. Compose the single spoken turn now."
        )
        # Orchestrator is the Gemini Flash provider (see config.AGENT_MODELS);
        # call it directly for a plain-text (non-JSON) response.
        result = await self.gemini.generate(
            model=self.model_name,
            prompt=prompt,
            system_instruction=self.system_prompt,
            temperature=self.temperature,
        )
        return result or ""
