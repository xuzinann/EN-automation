# W&B Weave Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Weights & Biases **Weave** observability + an evaluation harness to the Expert Call Agent so every agent turn, LLM call, and queue decision is traced, and flag-quality is measurable — targeting the hackathon's *Best Use of Weave* prize and the *Agent Orchestration / Technical Execution* criteria.

**Architecture:** Weave is added as an **optional, no-op-when-absent** layer. A tiny `observability.py` module wraps `weave.init()` and exposes an `op` decorator that falls back to identity when Weave isn't installed/enabled, so the app still runs without a W&B account. We decorate the client methods, `BaseAgent._call_model`, and each agent's `run()` so Weave reconstructs the full multi-agent harness as a nested trace tree. A separate `evals/` package scores flag quality with `weave.Evaluation`.

**Tech Stack:** Python 3.12, FastAPI, `google-genai` (Gemini on Vertex), `httpx` (Claude on Vertex `:rawPredict`), **`weave`** (new), **`pytest`** (new, dev/test).

---

## Background for the engineer (zero-context assumed)

- The app lives in `expert_call_agent/`. It runs via `python run.py` (uvicorn on port 8888). All imports are rooted at `expert_call_agent/` (see `sys.path.insert` in `run.py` and `api/app.py`), so **import paths are top-level** (e.g. `from clients.gemini_client import GeminiClient`, `from models import CallSession`).
- LLM calls funnel through two clients: `clients/gemini_client.py` (`GeminiClient.generate`, `.generate_with_audio`) and `clients/claude_client.py` (`ClaudeClient.generate`). Claude does **not** use the Anthropic SDK — it's raw `httpx` to Vertex — so Weave's SDK auto-integrations won't catch it. That's exactly why we use the `@op()` decorator (works on any function).
- All agents subclass `agents/base_agent.py::BaseAgent`. Every agent calls `self._call_model(...)`. Each agent implements `async def run(self, session, **kwargs)`.
- The live-call orchestration is in `api/routes_call.py` — it fans out agents with `asyncio.gather(...)`, then a deterministic queue (in `agents/orchestrator.py`) selects one action.
- **Auth:** Weave/W&B needs a W&B API key. Vertex (Gemini/Claude) uses ADC (already configured). The eval runner makes real Vertex calls, so it must run in an environment with both `WANDB_API_KEY` and working ADC.
- There is **no `tests/` directory and no pytest yet** — Task 1 adds pytest. Tests in this plan avoid `pytest-asyncio` by calling `asyncio.run(...)` inside the test, so no extra plugin is needed.
- `.gitignore` already ignores `.env`, `*.mp3`, `*.wav`, `__pycache__/`, `.claude/`.

> **Weave API note:** API shapes below (`weave.init`, `weave.op`, `weave.Evaluation`) match Weave ≥ 0.51. If a signature differs in the installed version, check https://weave-docs.wandb.ai — the structure of this plan does not change.

---

## File Structure

**New files**
- `expert_call_agent/observability.py` — `init_weave()`, `weave_enabled()`, and the `op` decorator. Single responsibility: make Weave optional and centralized.
- `expert_call_agent/evals/__init__.py` — marks the package.
- `expert_call_agent/evals/scorers.py` — pure scorer functions (no I/O, fully unit-testable).
- `expert_call_agent/evals/flag_dataset.json` — small hand-labeled eval dataset.
- `expert_call_agent/evals/run_flag_eval.py` — builds + runs the `weave.Evaluation`.
- `expert_call_agent/tests/__init__.py`
- `expert_call_agent/tests/test_observability.py` — tests the `op` passthrough + `init_weave` guard.
- `expert_call_agent/tests/test_scorers.py` — tests the scorers.
- `expert_call_agent/.env.example` — documents required env vars.

**Modified files**
- `expert_call_agent/requirements.txt` — add `weave`, `pytest`.
- `expert_call_agent/api/app.py` — call `init_weave()` on startup.
- `expert_call_agent/clients/gemini_client.py` — `@op()` on `generate`, `generate_with_audio`.
- `expert_call_agent/clients/claude_client.py` — `@op()` on `generate`.
- `expert_call_agent/clients/stt_client.py` — `@op()` on `transcribe`.
- `expert_call_agent/agents/base_agent.py` — `@op()` on `_call_model`.
- `expert_call_agent/agents/{qa_agent,followup_agent,note_taker,orchestrator,summarizer,context_ingestion,call_guide_drafter,post_call_summarizer}.py` — `@op()` on each `run`.

---

## Task 1: Add dependencies

**Files:**
- Modify: `expert_call_agent/requirements.txt`

- [ ] **Step 1: Add weave and pytest to requirements**

Append these two lines to `expert_call_agent/requirements.txt`:

```
weave>=0.51.0
pytest>=8.0.0
```

- [ ] **Step 2: Install**

Run (from `expert_call_agent/`):
```bash
pip install -r requirements.txt
```
Expected: installs `weave`, `pytest`, and their deps with no errors.

- [ ] **Step 3: Verify weave imports**

Run:
```bash
python -c "import weave, pytest; print('weave', weave.__version__)"
```
Expected: prints a weave version (e.g. `weave 0.51.x`), no traceback.

- [ ] **Step 4: Commit**

```bash
git add expert_call_agent/requirements.txt
git commit -m "chore: add weave and pytest dependencies"
```

---

## Task 2: Optional observability module (`op` decorator + init)

This module is the seam that makes Weave optional. Everything else imports `op` and `init_weave` from here.

**Files:**
- Create: `expert_call_agent/observability.py`
- Test: `expert_call_agent/tests/__init__.py`, `expert_call_agent/tests/test_observability.py`

- [ ] **Step 1: Write the failing test**

Create `expert_call_agent/tests/__init__.py` (empty file).

Create `expert_call_agent/tests/test_observability.py`:

```python
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import observability


def test_op_preserves_sync_return():
    @observability.op()
    def add(a, b):
        return a + b
    assert add(2, 3) == 5


def test_op_preserves_async_return():
    @observability.op()
    async def add(a, b):
        return a + b
    assert asyncio.run(add(2, 3)) == 5


def test_op_preserves_function_name():
    @observability.op()
    def my_func():
        return 1
    assert my_func.__name__ == "my_func"


def test_weave_disabled_env_disables(monkeypatch):
    monkeypatch.setenv("WEAVE_DISABLED", "1")
    assert observability.weave_enabled() is False


def test_init_weave_noop_when_disabled(monkeypatch):
    monkeypatch.setenv("WEAVE_DISABLED", "1")
    # Must not raise and must not require network / API key.
    assert observability.init_weave() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `expert_call_agent/`):
```bash
pytest tests/test_observability.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'observability'`.

- [ ] **Step 3: Write the implementation**

Create `expert_call_agent/observability.py`:

```python
"""Optional Weave (W&B) observability.

Designed to be a no-op when weave is not installed or is explicitly disabled,
so the app and tests run without a W&B account. Set WEAVE_DISABLED=1 to turn
tracing off even when weave is installed.
"""
import os

try:
    import weave
    _WEAVE_INSTALLED = True
except ImportError:  # weave not installed — degrade gracefully
    weave = None
    _WEAVE_INSTALLED = False

_initialized = False


def weave_enabled() -> bool:
    """True only if weave is importable and not disabled via env."""
    disabled = os.getenv("WEAVE_DISABLED", "").lower() in ("1", "true", "yes")
    return _WEAVE_INSTALLED and not disabled


def init_weave() -> bool:
    """Initialize Weave once. Returns True if tracing is now active.

    Reads WEAVE_PROJECT (default 'http418-expert-call'). Safe to call multiple
    times. No-ops (returns False) when weave is unavailable or disabled.
    """
    global _initialized
    if _initialized:
        return True
    if not weave_enabled():
        return False
    project = os.getenv("WEAVE_PROJECT", "http418-expert-call")
    weave.init(project)
    _initialized = True
    return True


def op(func=None, **kwargs):
    """`weave.op` when available, else an identity decorator.

    Use with parentheses: `@op()`. Works on sync and async functions and on
    methods. When weave is not installed, returns the function unchanged.
    """
    def decorator(f):
        if _WEAVE_INSTALLED:
            return weave.op(**kwargs)(f)
        return f

    if func is not None and callable(func):
        return decorator(func)
    return decorator
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
pytest tests/test_observability.py -v
```
Expected: PASS (5 passed). The async/sync tests pass whether or not weave is installed because `op` preserves behavior either way.

- [ ] **Step 5: Commit**

```bash
git add expert_call_agent/observability.py expert_call_agent/tests/__init__.py expert_call_agent/tests/test_observability.py
git commit -m "feat: add optional Weave observability module"
```

---

## Task 3: Initialize Weave at app startup

**Files:**
- Modify: `expert_call_agent/api/app.py` (lifespan, around line 21-28)

- [ ] **Step 1: Import and call `init_weave` in the lifespan**

In `expert_call_agent/api/app.py`, add the import near the other top-level imports (after line 18):

```python
from observability import init_weave
```

Then make `init_weave()` the first line inside the `lifespan` function. Change:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.gemini = GeminiClient()
```

to:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_weave()  # starts Weave tracing if WANDB_API_KEY is set and weave installed
    app.state.gemini = GeminiClient()
```

- [ ] **Step 2: Verify the app still boots with tracing disabled**

Run (from `expert_call_agent/`):
```bash
WEAVE_DISABLED=1 python -c "from api.app import create_app; create_app(); print('app builds OK')"
```
Expected: prints `app builds OK`, no traceback. (Construction doesn't trigger lifespan, but this confirms imports are clean.)

- [ ] **Step 3: Commit**

```bash
git add expert_call_agent/api/app.py
git commit -m "feat: initialize Weave on app startup"
```

---

## Task 4: Trace the client layer (lowest-level Vertex calls)

**Files:**
- Modify: `expert_call_agent/clients/gemini_client.py`
- Modify: `expert_call_agent/clients/claude_client.py`
- Modify: `expert_call_agent/clients/stt_client.py`

- [ ] **Step 1: Decorate GeminiClient methods**

In `clients/gemini_client.py`, add the import at the top (after `import config`):
```python
from observability import op
```
Add `@op()` directly above both `async def generate(` and `async def generate_with_audio(`:
```python
    @op()
    async def generate(
        self,
        model: str,
        prompt: str,
        ...
```
```python
    @op()
    async def generate_with_audio(
        self,
        model: str,
        ...
```

- [ ] **Step 2: Decorate ClaudeClient.generate**

In `clients/claude_client.py`, add after `import config`:
```python
from observability import op
```
Add `@op()` above `async def generate(`:
```python
    @op()
    async def generate(
        self,
        messages: list[dict],
        ...
```

- [ ] **Step 3: Decorate STTClient.transcribe**

Open `clients/stt_client.py`. Add `from observability import op` to its imports, and add `@op()` directly above the `async def transcribe(` method.

- [ ] **Step 4: Smoke-test imports**

Run (from `expert_call_agent/`):
```bash
WEAVE_DISABLED=1 python -c "from clients.gemini_client import GeminiClient; from clients.claude_client import ClaudeClient; from clients.stt_client import STTClient; print('clients import OK')"
```
Expected: prints `clients import OK`.

- [ ] **Step 5: Commit**

```bash
git add expert_call_agent/clients/gemini_client.py expert_call_agent/clients/claude_client.py expert_call_agent/clients/stt_client.py
git commit -m "feat: trace client LLM/STT calls with Weave"
```

---

## Task 5: Trace the agent layer (builds the harness trace tree)

Decorating `_call_model` (one place) plus each agent's `run` produces a nested trace: `qa_agent.run → BaseAgent._call_model → ClaudeClient.generate`, etc.

**Files:**
- Modify: `expert_call_agent/agents/base_agent.py`
- Modify: `expert_call_agent/agents/qa_agent.py`
- Modify: `expert_call_agent/agents/followup_agent.py`
- Modify: `expert_call_agent/agents/note_taker.py`
- Modify: `expert_call_agent/agents/orchestrator.py`
- Modify: `expert_call_agent/agents/summarizer.py`
- Modify: `expert_call_agent/agents/context_ingestion.py`
- Modify: `expert_call_agent/agents/call_guide_drafter.py`
- Modify: `expert_call_agent/agents/post_call_summarizer.py`
- Test: `expert_call_agent/tests/test_tracing_smoke.py`

- [ ] **Step 1: Decorate `BaseAgent._call_model`**

In `agents/base_agent.py`, add `from observability import op` after `import config` (line 5), then add `@op()` above `async def _call_model(`:
```python
    @op()
    async def _call_model(self, user_prompt: str) -> str:
```

- [ ] **Step 2: Decorate every agent's `run`**

In **each** of these files, add `from observability import op` to the imports and put `@op()` on the line directly above its `async def run(`:
`qa_agent.py`, `followup_agent.py`, `note_taker.py`, `orchestrator.py`, `summarizer.py`, `context_ingestion.py`, `call_guide_drafter.py`, `post_call_summarizer.py`.

Example for `qa_agent.py` (the `run` is at line 23):
```python
    @op()
    async def run(self, session: CallSession, **kwargs) -> AgentAction:
```

> Note: some agents may define helper methods like `get_latest_notes` / `get_latest_coverage` / `get_latest_takeaways`. Do **not** decorate those — only `run`.

- [ ] **Step 3: Write a smoke test that the decorated agents still import and expose `run`**

Create `expert_call_agent/tests/test_tracing_smoke.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.qa_agent import QAAgent
from agents.followup_agent import FollowUpAgent
from agents.note_taker import NoteTakerAgent
from agents.orchestrator import OrchestratorAgent


def test_agents_still_have_callable_run():
    for cls in (QAAgent, FollowUpAgent, NoteTakerAgent, OrchestratorAgent):
        assert hasattr(cls, "run")
        assert callable(cls.run)
```

- [ ] **Step 4: Run the smoke test**

Run (from `expert_call_agent/`):
```bash
WEAVE_DISABLED=1 pytest tests/test_tracing_smoke.py -v
```
Expected: PASS (1 passed). Confirms the decorators didn't break class definitions.

- [ ] **Step 5: Commit**

```bash
git add expert_call_agent/agents/ expert_call_agent/tests/test_tracing_smoke.py
git commit -m "feat: trace agent.run and _call_model with Weave"
```

---

## Task 6: Eval scorers (pure, TDD)

**Files:**
- Create: `expert_call_agent/evals/__init__.py`
- Create: `expert_call_agent/evals/scorers.py`
- Test: `expert_call_agent/tests/test_scorers.py`

- [ ] **Step 1: Write the failing test**

Create `expert_call_agent/evals/__init__.py` (empty).

Create `expert_call_agent/tests/test_scorers.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.scorers import flag_type_match, produced_question


def test_flag_type_match_correct():
    assert flag_type_match("must_ask", {"flag_type": "must_ask"}) == {"correct": True}


def test_flag_type_match_wrong():
    assert flag_type_match("must_ask", {"flag_type": "nice_to_have"}) == {"correct": False}


def test_flag_type_match_missing_output():
    assert flag_type_match("must_ask", {}) == {"correct": False}
    assert flag_type_match("must_ask", None) == {"correct": False}


def test_produced_question_nonempty():
    assert produced_question({"content": "What is the GPU count?"}) == {"nonempty": True}


def test_produced_question_empty():
    assert produced_question({"content": "   "}) == {"nonempty": False}
    assert produced_question({}) == {"nonempty": False}
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `expert_call_agent/`):
```bash
pytest tests/test_scorers.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'evals.scorers'`.

- [ ] **Step 3: Write the implementation**

Create `expert_call_agent/evals/scorers.py`:

```python
"""Pure scoring functions for Weave evaluations.

Each scorer takes dataset columns (by name) plus the model `output` dict and
returns a dict of metrics. Kept free of I/O so they are unit-testable.
"""


def flag_type_match(expected_flag: str, output: dict) -> dict:
    """Did the agent assign the expected flag tier?"""
    got = (output or {}).get("flag_type")
    return {"correct": got == expected_flag}


def produced_question(output: dict) -> dict:
    """Did the agent actually produce a non-empty question?"""
    content = (output or {}).get("content", "")
    return {"nonempty": bool(content and content.strip())}
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
pytest tests/test_scorers.py -v
```
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add expert_call_agent/evals/__init__.py expert_call_agent/evals/scorers.py expert_call_agent/tests/test_scorers.py
git commit -m "feat: add Weave eval scorers with tests"
```

---

## Task 7: Eval dataset

**Files:**
- Create: `expert_call_agent/evals/flag_dataset.json`

- [ ] **Step 1: Create the dataset**

Create `expert_call_agent/evals/flag_dataset.json`. Each row is one expert utterance plus the flag tier a good QA agent should produce next. (Rows are seeded from the CoreWeave demo in `api/routes_call.py`; expand later.)

```json
[
  {
    "transcript": "[expert] When I left we had roughly 40,000 H100s deployed across six data centers, with another 25,000 on order.",
    "expected_flag": "must_ask"
  },
  {
    "transcript": "[expert] On paper you see 60 percent gross margin, but real margin on deployed capacity was closer to 42 to 45 percent.",
    "expected_flag": "must_ask"
  },
  {
    "transcript": "[expert] Microsoft alone was roughly 35 percent of revenue, and the top five customers combined were about 62 percent.",
    "expected_flag": "must_ask"
  },
  {
    "transcript": "[expert] Thanks for having me. I was the VP of Infrastructure at CoreWeave for about two years.",
    "expected_flag": "should_ask"
  },
  {
    "transcript": "[expert] Power is becoming the real bottleneck, not GPUs. We paid 4.5 to 6 cents per kilowatt hour on our best PPAs.",
    "expected_flag": "should_ask"
  }
]
```

> Labels are a starting hypothesis (numbers/contradictions → `must_ask`; context/color → `should_ask`). The point of the eval is to measure how often the agent agrees; adjust labels as the team aligns on what "good" looks like.

- [ ] **Step 2: Validate the JSON parses**

Run (from `expert_call_agent/`):
```bash
python -c "import json; rows=json.load(open('evals/flag_dataset.json')); print(len(rows), 'rows'); assert all('transcript' in r and 'expected_flag' in r for r in rows)"
```
Expected: prints `5 rows`, no assertion error.

- [ ] **Step 3: Commit**

```bash
git add expert_call_agent/evals/flag_dataset.json
git commit -m "feat: add flag-quality eval dataset"
```

---

## Task 8: Eval runner (`weave.Evaluation`)

**Files:**
- Create: `expert_call_agent/evals/run_flag_eval.py`

> Requires `WANDB_API_KEY` set and working Vertex ADC, because it makes real Gemini/Claude calls. The QA agent uses Claude by default (see `config.AGENT_MODELS`).

- [ ] **Step 1: Write the runner**

Create `expert_call_agent/evals/run_flag_eval.py`:

```python
"""Run a Weave evaluation of the QA agent's flag quality.

Usage (from expert_call_agent/):
    WANDB_API_KEY=... python -m evals.run_flag_eval

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
        return action.model_dump()

    evaluation = weave.Evaluation(
        dataset=_load_dataset(),
        scorers=[flag_type_match, produced_question],
    )
    await evaluation.evaluate(predict)
    await claude.close()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Syntax / import check (no network)**

Run (from `expert_call_agent/`):
```bash
WEAVE_DISABLED=1 python -c "import ast; ast.parse(open('evals/run_flag_eval.py').read()); print('parses OK')"
```
Expected: prints `parses OK`.

- [ ] **Step 3: Run the real evaluation**

Ensure `WANDB_API_KEY` is set (get it from https://wandb.ai/authorize) and ADC works. Run (from `expert_call_agent/`):
```bash
WANDB_API_KEY=<your-key> python -m evals.run_flag_eval
```
Expected: Weave prints a project URL and an evaluation summary table with `flag_type_match.correct` and `produced_question.nonempty` means. Open the URL → confirm a `predict` trace per row, each with nested `qa_agent.run → _call_model → ClaudeClient.generate` spans.

> If `weave.Evaluation.evaluate(predict)` rejects a bare function in the installed version, wrap it in a `weave.Model` subclass with an `@weave.op() async def predict(self, transcript)` method and pass the instance instead. See https://weave-docs.wandb.ai/guides/core-types/evaluations.

- [ ] **Step 4: Commit**

```bash
git add expert_call_agent/evals/run_flag_eval.py
git commit -m "feat: add Weave evaluation runner for QA flag quality"
```

---

## Task 9: Document env vars + how to run

**Files:**
- Create: `expert_call_agent/.env.example`
- Modify: `expert_call_agent/requirements.txt` is already done; no change here.

- [ ] **Step 1: Create `.env.example`**

Create `expert_call_agent/.env.example`:

```bash
# Weave / W&B observability
# Get a key at https://wandb.ai/authorize
WANDB_API_KEY=
# Weave project name (optional; defaults to http418-expert-call)
WEAVE_PROJECT=http418-expert-call
# Set to 1 to fully disable Weave tracing (app still runs)
WEAVE_DISABLED=
```

- [ ] **Step 2: Verify .env is gitignored (it already is)**

Run (from repo root):
```bash
git check-ignore expert_call_agent/.env || echo "WARNING: .env not ignored"
```
Expected: prints the path `expert_call_agent/.env` (meaning it IS ignored). `.env.example` is **not** ignored and should be committed.

- [ ] **Step 3: Run the full test suite**

Run (from `expert_call_agent/`):
```bash
WEAVE_DISABLED=1 pytest -v
```
Expected: all tests pass (observability + scorers + tracing smoke).

- [ ] **Step 4: Commit**

```bash
git add expert_call_agent/.env.example
git commit -m "docs: document Weave env vars in .env.example"
```

---

## Done / Demo checklist

- [ ] `WEAVE_DISABLED=1 pytest -v` → all green (works with no W&B account).
- [ ] `WANDB_API_KEY=... python run.py`, run the demo flow (`run_demo` over the websocket / the demo button in `static/index.html`) → Weave UI shows a nested trace per transcript turn: parallel `qa/followup/note_taker/summarizer` runs → orchestrator selection → TTS.
- [ ] `WANDB_API_KEY=... python -m evals.run_flag_eval` → Weave shows an evaluation with flag-quality scores.
- [ ] Screenshot the trace tree + the eval table for the pitch/demo.

## How this maps to judging

- **Agent Orchestration / Technical Execution:** the Weave trace tree literally visualizes multiple agents running in parallel and the deterministic queue selecting one — proof, not claims.
- **Best Use of Weave:** tracing *and* an evaluation harness with custom scorers (not just autologging).
- **Sponsor Usage:** Weave (W&B) used meaningfully end-to-end, alongside Gemini/Claude on Vertex.

## Notes / follow-ups (out of scope for this plan)

- Add an LLM-as-judge scorer for **note factuality** (compare `note_taker` output against the source utterance).
- Add a `followup_agent` contradiction-recall eval (dataset of utterances that conflict with known data points).
- Consider a `weave.Model` subclass so model config (provider/model name) is versioned in Weave.
