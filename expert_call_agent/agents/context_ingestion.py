from agents.base_agent import BaseAgent
from models import CallSession, ProjectContext


class ContextIngestionAgent(BaseAgent):
    name = "context_ingestion"
    system_prompt = (
        "You are an expert PE due diligence analyst. Given a project brief, "
        "extract structured information and return ONLY valid JSON with these fields:\n"
        "{\n"
        '  "company_name": "string",\n'
        '  "industry": "string",\n'
        '  "deal_type": "string (e.g. acquisition, growth equity, buyout)",\n'
        '  "key_hypotheses": ["strings"],\n'
        '  "known_data_points": [{"category": "string", "description": "string", '
        '"value": "string or null", "confidence": "stated|estimated|unknown", "source": "string"}],\n'
        '  "gaps_to_fill": ["strings — questions that need answering via expert calls"],\n'
        '  "key_entities": [{"name": "string", "role": "string", "relevance": "string"}],\n'
        '  "expert_profile": "string — description of ideal expert to interview"\n'
        "}\n"
        "Return ONLY the JSON object, no markdown fences or commentary."
    )

    async def run(self, session: CallSession, brief_text: str = "") -> ProjectContext:
        prompt = f"Analyze this PE due diligence project brief and extract structured context:\n\n{brief_text}"
        response = await self._call_model(prompt)
        data = self._parse_json(response)
        return ProjectContext.model_validate(data)
