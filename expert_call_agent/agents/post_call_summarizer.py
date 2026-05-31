import json
import re
from agents.base_agent import BaseAgent
from models import CallSession
from observability import op


class PostCallSummarizerAgent(BaseAgent):
    name = "post_call_summarizer"
    system_prompt = (
        "Generate a post-call deliverable for PE due diligence in TWO tagged parts.\n\n"
        "PART 1 — between <TAKEAWAYS> and </TAKEAWAYS>, output ONLY valid JSON:\n"
        '{"takeaways": ["top findings"], "surprises": ["deviations from the client\'s '
        'hypotheses"], "remaining_gaps": ["open questions still unanswered"]}\n\n'
        "PART 2 — between <REPORT> and </REPORT>, output a markdown report with these sections:\n"
        "# Expert Call Summary\n## Executive Summary\n## Key Findings by Category\n"
        "### Headcount & Org Structure\n### Operations & Technology\n### Market & Clients\n"
        "### Financials\n## Data Points Confirmed (table: Data Point | Value | Confidence | Source)\n"
        "## Gaps Remaining\n## Contradictions & Red Flags\n## Expert Credibility Assessment\n"
        "## Recommended Follow-Up Actions\n\n"
        "Use actual numbers and quotes from the call. Output nothing outside the two tagged blocks."
    )

    @op()
    async def run(self, session: CallSession, **kwargs) -> dict:
        transcript = "\n".join(f"[{e.speaker}] {e.text}" for e in session.transcript)
        notes = json.dumps([n.model_dump() for n in session.notes])
        context = session.project_context.model_dump_json() if session.project_context else "{}"

        prompt = (
            f"Project context:\n{context}\n\n"
            f"Full transcript:\n{transcript}\n\n"
            f"Structured notes:\n{notes}\n\n"
            "Generate the two-part deliverable."
        )

        if self.model_provider == "claude":
            raw = await self.claude.generate(
                messages=[{"role": "user", "content": prompt}],
                system=self.system_prompt,
                max_tokens=8192,
            )
        else:
            raw = await self.gemini.generate(
                model=self.model_name,
                prompt=prompt,
                system_instruction=self.system_prompt,
            )
        return self._parse_deliverable(raw)

    def _parse_deliverable(self, raw: str) -> dict:
        result = {"markdown": raw.strip(), "takeaways": [], "surprises": [], "remaining_gaps": []}

        t = re.search(r"<TAKEAWAYS>(.*?)</TAKEAWAYS>", raw, re.DOTALL)
        if t:
            block = re.sub(r"^```(?:json)?|```$", "", t.group(1).strip(), flags=re.MULTILINE).strip()
            try:
                data = json.loads(block)
                result["takeaways"] = data.get("takeaways", [])
                result["surprises"] = data.get("surprises", [])
                result["remaining_gaps"] = data.get("remaining_gaps", [])
            except json.JSONDecodeError:
                pass

        r = re.search(r"<REPORT>(.*?)</REPORT>", raw, re.DOTALL)
        if r:
            result["markdown"] = r.group(1).strip()
        return result
