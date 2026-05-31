"""Run a Weave evaluation of the QA agent's flag quality.

Usage (from expert_call_agent/):
    WANDB_API_KEY=... python3 -m evals.run_flag_eval

Logs traces + scores to the Weave project (WEAVE_PROJECT, default
'http418-expert-call'). Open the URL printed by weave.init to inspect.
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import weave

from observability import init_weave
from clients.gemini_client import GeminiClient
from clients.claude_client import ClaudeClient
from agents.qa_agent import QAAgent
from models import CallSession, TranscriptEntry
from evals.scorers import flag_type_match, produced_question


def _load_dataset() -> list[dict]:
    path = Path(__file__).resolve().parent / "flag_dataset.json"
    return json.loads(path.read_text())


def _build_session(transcript: str) -> CallSession:
    """Minimal live session containing a single expert utterance."""
    session = CallSession(status="live")
    # transcript lines look like "[expert] ..."; strip the speaker tag.
    text = transcript.split("] ", 1)[-1] if transcript.startswith("[") else transcript
    session.transcript = [TranscriptEntry(speaker="expert", text=text)]
    return session


async def main() -> None:
    if not init_weave():
        raise SystemExit(
            "Weave is not enabled. Install weave and set WANDB_API_KEY "
            "(and ensure WEAVE_DISABLED is unset)."
        )

    gemini = GeminiClient()
    claude = ClaudeClient()
    qa = QAAgent(gemini, claude)

    @weave.op()
    async def predict(transcript: str) -> dict:
        action = await qa.run(_build_session(transcript))
        # QA returns None when it chooses to stay silent; normalize for scorers.
        return action.model_dump() if action is not None else {}

    evaluation = weave.Evaluation(
        dataset=_load_dataset(),
        scorers=[flag_type_match, produced_question],
    )
    await evaluation.evaluate(predict)
    await claude.close()


if __name__ == "__main__":
    asyncio.run(main())
