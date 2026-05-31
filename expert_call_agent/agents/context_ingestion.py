from agents.base_agent import BaseAgent
from models import CallSession, ProjectContext


class ContextIngestionAgent(BaseAgent):
    name = "context_ingestion"
    system_prompt = (
        "You are an expert PE due diligence analyst. Given a project brief, "
        "extract structured information and return ONLY valid JSON with these fields:\n"
        "{\n"
        '  "company_name": "string — the primary target company being evaluated '
        '(e.g. if studying CoreWeave\'s competitive landscape, company_name is CoreWeave)",\n'
        '  "industry": "string — the industry sector",\n'
        '  "deal_type": "string — the PE deal stage. MUST be one of: '
        "early stage, growth equity, buyout, carve-out, turnaround, "
        'secondary, add-on, platform, recapitalization, or other PE stage",\n'
        '  "study_type": "string — the type of research, e.g.: '
        "competitive landscape, commercial due diligence, market sizing, "
        'operational DD, vendor DD, technology assessment",\n'
        '  "num_calls": "integer — total number of expert calls planned (N)",\n'
        '  "call_targets": ["strings — list of companies/organizations to call, in order"],\n'
        '  "key_hypotheses": ["strings"],\n'
        '  "known_data_points": [{"category": "string", "description": "string", '
        '"value": "string or null", "confidence": "stated|estimated|unknown", "source": "string"}],\n'
        '  "gaps_to_fill": ["strings — questions that need answering via expert calls"],\n'
        '  "key_entities": [{"name": "string", "role": "string", "relevance": "string"}],\n'
        '  "expert_profile": "string — description of ideal experts to interview"\n'
        "}\n"
        "Return ONLY the JSON object, no markdown fences or commentary."
    )

    async def run(self, session: CallSession, brief_text: str = "") -> ProjectContext:
        prompt = f"Analyze this PE due diligence project brief and extract structured context:\n\n{brief_text}"
        response = await self._call_model(prompt)
        data = self._parse_json(response)
        return ProjectContext.model_validate(data)
