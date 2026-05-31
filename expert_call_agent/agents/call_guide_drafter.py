from agents.base_agent import BaseAgent
from models import CallSession, CallGuide


class CallGuideDrafterAgent(BaseAgent):
    name = "call_guide_drafter"
    system_prompt = (
        "You are a senior PE consultant preparing for an expert network call. "
        "Given structured project context, create a comprehensive interview guide. "
        "Return ONLY valid JSON with these fields:\n"
        "{\n"
        '  "opening_questions": [{"text": "string", "rationale": "string", '
        '"priority": "must_ask|should_ask|nice_to_have", "target_data": ["strings"]}],\n'
        '  "deep_dive_sections": [{"topic": "string", "questions": [same format], '
        '"coverage_threshold": "string describing what covered means"}],\n'
        '  "red_flag_probes": [same question format],\n'
        '  "data_points_to_confirm": ["strings"],\n'
        '  "closing_questions": [same question format]\n'
        "}\n"
        "Include sections for: Org Structure, Headcount by Role, Operations, "
        "Market & Competition, Technology, and Key Risks.\n"
        "Return ONLY the JSON object, no markdown fences or commentary."
    )

    async def run(self, session: CallSession, **kwargs) -> CallGuide:
        context_json = session.project_context.model_dump_json() if session.project_context else "{}"
        prompt = f"Create an interview guide based on this project context:\n\n{context_json}"
        response = await self._call_model(prompt)
        data = self._parse_json(response)
        return CallGuide.model_validate(data)
