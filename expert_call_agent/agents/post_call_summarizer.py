import json
from agents.base_agent import BaseAgent
from models import CallSession


class PostCallSummarizerAgent(BaseAgent):
    name = "post_call_summarizer"
    system_prompt = (
        "Generate a comprehensive post-call summary for PE due diligence. "
        "Use markdown formatting. Structure:\n\n"
        "# Expert Call Summary\n"
        "## Executive Summary\n"
        "3-5 sentences capturing the most important findings.\n\n"
        "## Key Findings by Category\n"
        "### Headcount & Org Structure\n"
        "### Operations & Technology\n"
        "### Market & Clients\n"
        "### Financials\n\n"
        "## Data Points Confirmed\n"
        "Table with: Data Point | Value | Confidence | Source\n\n"
        "## Gaps Remaining\n"
        "What we still don't know.\n\n"
        "## Contradictions & Red Flags\n\n"
        "## Expert Credibility Assessment\n"
        "How credible was this expert? What was their visibility level?\n\n"
        "## Recommended Follow-Up Actions\n"
        "Numbered list of next steps.\n\n"
        "Be thorough and specific. Use actual numbers and quotes from the call."
    )

    async def run(self, session: CallSession, **kwargs) -> str:
        transcript = "\n".join(
            f"[{e.speaker}] {e.text}" for e in session.transcript
        )
        notes = json.dumps([n.model_dump() for n in session.notes])
        takeaways = json.dumps(session.key_takeaways)
        context = session.project_context.model_dump_json() if session.project_context else "{}"

        prompt = (
            f"Project context:\n{context}\n\n"
            f"Full transcript:\n{transcript}\n\n"
            f"Structured notes:\n{notes}\n\n"
            f"Key takeaways identified during call:\n{takeaways}\n\n"
            "Generate the comprehensive post-call summary."
        )

        # Use text response, not JSON
        if self.model_provider == "claude":
            return await self.claude.generate(
                messages=[{"role": "user", "content": prompt}],
                system=self.system_prompt,
            )
        else:
            return await self.gemini.generate(
                model=self.model_name,
                prompt=prompt,
                system_instruction=self.system_prompt,
            )
