import json
from agents.base_agent import BaseAgent
from models import CallSession, AgentAction
import config


class OrchestratorAgent(BaseAgent):
    name = "orchestrator"
    system_prompt = (
        "You are the call moderator for a PE expert interview. "
        "You receive proposed actions from multiple specialist agents. Your job:\n"
        "1. Select the single best next action (question, follow-up, etc.)\n"
        "2. Prevent redundant questions\n"
        "3. Manage pacing — don't rush the expert\n"
        "4. Prioritize must-ask questions over nice-to-haves\n"
        "5. Consider what was just discussed before suggesting the next question\n\n"
        "Return ONLY valid JSON:\n"
        '{"selected_index": 0, "reason": "why you chose this action", '
        '"defer_indices": [1, 2]}\n'
        "selected_index is the 0-based index of the best action from the list.\n"
        "No markdown fences or commentary."
    )

    async def run(self, session: CallSession, **kwargs) -> dict:
        pending = session.pending_actions
        if not pending:
            return {"selected_action": None, "reason": "no pending actions"}

        recent = session.transcript[-10:]
        transcript = "\n".join(f"[{e.speaker}] {e.text}" for e in recent)

        actions_list = []
        for i, a in enumerate(pending):
            actions_list.append({
                "index": i,
                "agent": a.agent_name,
                "type": a.action_type,
                "content": a.content,
                "priority": a.priority,
            })

        prompt = (
            f"Recent transcript:\n{transcript}\n\n"
            f"Pending agent actions:\n{json.dumps(actions_list, indent=2)}\n\n"
            "Select the best action to surface to the interviewer."
        )
        response = await self._call_model(prompt)
        parsed = self._parse_json(response)

        idx = parsed.get("selected_index", 0)
        if 0 <= idx < len(pending):
            selected = pending[idx]
            return {
                "selected_action": selected.model_dump(),
                "reason": parsed.get("reason", ""),
            }

        return {
            "selected_action": pending[0].model_dump() if pending else None,
            "reason": parsed.get("reason", "fallback to first action"),
        }
