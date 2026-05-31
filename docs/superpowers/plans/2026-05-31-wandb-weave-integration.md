# W&B Weave Observability + Eval Harness — Implementation Plan (reconciled)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Weights & Biases **Weave** tracing + a small evaluation harness to the Expert Call Agent so every agent turn, LLM/STT call, and the deterministic queue decision is observable, and QA flag-quality is measurable — targeting the hackathon's *Best Use of Weave* prize and the *Agent Orchestration / Technical Execution* criteria.

**Architecture:** Weave is an **optional, no-op-when-absent** layer. A tiny `observability.py` exposes `init_weave()` + an `op` decorator that degrades to identity when Weave is missing/disabled, so the app still runs with no W&B account. We decorate the two client classes, `BaseAgent._call_model` **and** `_call_model_text`, and each agent's `run()`; Weave then reconstructs each agent call as a nested trace (`agent.run → _call_model → client.generate`). A separate `evals/` package scores flag quality with `weave.Evaluation` over a small hand-labeled dataset.

**Tech Stack:** Python 3.12, FastAPI, `google-genai` (Gemini on Vertex), `httpx` (Claude + STT + TTS on Vertex/Cloud, raw REST), **`weave`** (new).

---

## Decisions baked into this plan (confirmed with the user)

1. **W&B key is available.** A `WANDB_API_KEY` will be provided (free from https://wandb.ai/authorize). The plan therefore ends with a *real* trace + eval run, not just a disabled smoke.
2. **Lightweight smoke, not TDD.** Per the repo's hackathon posture, **no `pytest` dependency.** Verification is import/parse smokes (`WEAVE_DISABLED=1 python3 -c …`) plus a one-line inline scorer assert. (This is the one deliberate divergence from the older draft of this plan.)
3. **Agents/clients only — `routes_call.py` is NOT touched.** We do not add a per-turn root span. Expected trace shape: during each expert turn the three agents run via `asyncio.gather`, so Weave records **three sibling trace trees** (`qa_agent.run`, `followup_agent.run`, `note_taker.run`), each nesting its `_call_model → client.generate`; the spoken turn is a fourth tree (`orchestrator.run → _call_model_text → claude.generate`). Pre-call shows `context_ingestion.run` / `call_guide_drafter.run`; post-call shows `post_call_summarizer.run`. This keeps the live hot path untouched.

---

## Why the previous draft of this file was rewritten (context, not steps)

The earlier version predated the deterministic-orchestration refactor and was stale: it decorated `agents/summarizer.py` (since **deleted**), missed `_call_model_text` (so the **orchestrator's** LLM call would be invisible), assumed the queue lived in `orchestrator.py` (it's now `live_orchestration.py::LiveCallQueue`), traced only one STT client (there are now two), and used port 8888 + a `pip install` that PEP 668 blocks on this VM. All corrected below.

---

## Project conventions (read before starting)

- Work **directly on `main`** — no feature branches.
- `python3` only (`python` is not on PATH). All commands assume cwd `/home/kevin/EN-automation/expert_call_agent` unless stated.
- The VM is **PEP 668 externally-managed**; installs MUST use `python3 -m pip install --user --break-system-packages …` (plain `--user` is blocked; deps live in `~/.local`).
- The server runs on **`127.0.0.1:8899`** (NOT 8888 — another user's copy of this app is on 8888 on this shared VM). `run.py`'s default is 8888, so launch the server explicitly on 8899 for any smoke (see Task 10).
- Imports are top-level rooted at `expert_call_agent/` (see `sys.path.insert` in `run.py`/`api/app.py`), e.g. `from clients.claude_client import ClaudeClient`, `import config`.
- Commit message trailer (every commit): `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `expert_call_agent/requirements.txt` | Declare `weave` | Modify (+1 line) |
| `expert_call_agent/observability.py` | The optional-Weave seam: `op`, `init_weave`, `weave_enabled` | **Create** |
| `expert_call_agent/api/app.py` | Start tracing on boot | Modify (import + 1 line in lifespan) |
| `expert_call_agent/clients/claude_client.py` | Trace `generate` | Modify (import + `@op()`) |
| `expert_call_agent/clients/gemini_client.py` | Trace `generate`, `generate_with_audio` | Modify (import + 2× `@op()`) |
| `expert_call_agent/clients/cloud_stt_client.py` | Trace `transcribe` (Chirp 2, default STT) | Modify (import + `@op()`) |
| `expert_call_agent/clients/stt_client.py` | Trace `transcribe` (Gemini STT fallback) | Modify (import + `@op()`) |
| `expert_call_agent/agents/base_agent.py` | Trace `_call_model` + `_call_model_text` | Modify (import + 2× `@op()`) |
| `expert_call_agent/agents/{qa_agent,followup_agent,note_taker,orchestrator,context_ingestion,call_guide_drafter,post_call_summarizer}.py` | Trace each `run` | Modify (import + `@op()`) ×7 |
| `expert_call_agent/evals/__init__.py` | Package marker | **Create** |
| `expert_call_agent/evals/scorers.py` | Pure scorer functions | **Create** |
| `expert_call_agent/evals/flag_dataset.json` | Hand-labeled QA eval dataset | **Create** |
| `expert_call_agent/evals/run_flag_eval.py` | `weave.Evaluation` runner | **Create** |
| `expert_call_agent/.env.example` | Document env vars | **Create** |

> **Not decorated, on purpose:** `tts_client.synthesize` (returns raw audio `bytes` — Weave would log large/noisy binary), and the agents' `get_latest_*` helper getters (only `run` is the meaningful unit). The `LiveCallQueue` is pure deterministic Python with no LLM call — left untraced because (a) the "agents/clients only" decision keeps us out of the hot path, and (b) it would need `routes_call.py` wiring to appear as a span.

---

## Task 1: Declare and install `weave`

**Files:**
- Modify: `expert_call_agent/requirements.txt`

- [ ] **Step 1: Append weave to requirements**

Append this single line to the end of `expert_call_agent/requirements.txt` (no `pytest` — lightweight posture):

```
weave>=0.51.0
```

- [ ] **Step 2: Install (PEP 668 — note the flags)**

Run (from `expert_call_agent/`):
```bash
python3 -m pip install --user --break-system-packages "weave>=0.51.0"
```
Expected: installs `weave` and its deps (pulls a fair amount, incl. its own client libs) into `~/.local`, no errors.

- [ ] **Step 3: Verify weave imports**

Run:
```bash
python3 -c "import weave; print('weave', weave.__version__)"
```
Expected: prints a weave version (e.g. `weave 0.51.x`), no traceback.

- [ ] **Step 4: Commit**

```bash
git add expert_call_agent/requirements.txt
git commit -m "chore: add weave dependency

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: The optional observability seam (`observability.py`)

This module is the single seam that makes Weave optional. Everything else imports `op` / `init_weave` from here. It improves on the older draft in two ways: `op` degrades to identity when **disabled** (not just when uninstalled), and `init_weave` swallows a failed `weave.init` (e.g. missing key) so the app never crashes on boot.

**Files:**
- Create: `expert_call_agent/observability.py`

- [ ] **Step 1: Create the module**

Create `expert_call_agent/observability.py` with exactly:

```python
"""Optional Weave (W&B) observability.

A no-op when weave is not installed or is explicitly disabled, so the app runs
without a W&B account. Set WEAVE_DISABLED=1 to turn tracing off even when weave
is installed (e.g. when you have no WANDB_API_KEY).
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

    Reads WEAVE_PROJECT (default 'http418-expert-call'). Safe to call repeatedly.
    No-ops (returns False) when weave is unavailable/disabled, and never raises —
    a failed init (e.g. no WANDB_API_KEY) just disables tracing instead of
    crashing app startup.
    """
    global _initialized
    if _initialized:
        return True
    if not weave_enabled():
        return False
    project = os.getenv("WEAVE_PROJECT", "http418-expert-call")
    try:
        weave.init(project)
    except Exception as e:  # missing key, network, etc. — degrade, don't crash
        print(f"[observability] weave.init failed ({e!r}); tracing disabled")
        return False
    _initialized = True
    return True


def op(func=None, **kwargs):
    """`weave.op` when tracing is enabled, else an identity decorator.

    Use with parentheses: `@op()`. Works on sync and async functions and on
    methods. When weave is missing OR WEAVE_DISABLED is set, returns the function
    unchanged (zero weave involvement). Decoration happens at import time, so set
    WEAVE_DISABLED before importing app modules for a clean no-op.
    """
    def decorator(f):
        if weave_enabled():
            return weave.op(**kwargs)(f)
        return f

    if func is not None and callable(func):
        return decorator(func)
    return decorator
```

- [ ] **Step 2: Smoke — passthrough works disabled AND enabled-ish**

Run (from `expert_call_agent/`):
```bash
WEAVE_DISABLED=1 python3 -c "
import observability as o
@o.op()
def add(a, b): return a + b
@o.op()
async def aadd(a, b): return a + b
import asyncio
assert add(2, 3) == 5
assert asyncio.run(aadd(2, 3)) == 5
assert add.__name__ == 'add'
assert o.weave_enabled() is False
assert o.init_weave() is False
print('observability OK (disabled)')
"
```
Expected: prints `observability OK (disabled)`.

- [ ] **Step 3: Commit**

```bash
git add expert_call_agent/observability.py
git commit -m "feat: optional Weave observability seam (op + init_weave)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Start tracing on app boot

**Files:**
- Modify: `expert_call_agent/api/app.py`

- [ ] **Step 1: Import `init_weave`**

In `api/app.py`, after the existing `import config` (line 11), add:

```python
from observability import init_weave
```

- [ ] **Step 2: Call it first in the lifespan**

In `api/app.py`, the lifespan currently begins:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.gemini = GeminiClient()
```

Change it to:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_weave()  # starts Weave tracing if enabled (WANDB_API_KEY set, weave installed)
    app.state.gemini = GeminiClient()
```

- [ ] **Step 3: Verify the app still builds with tracing disabled**

Run (from `expert_call_agent/`):
```bash
WEAVE_DISABLED=1 python3 -c "from api.app import create_app; create_app(); print('app builds OK')"
```
Expected: prints `app builds OK`, no traceback. (Construction doesn't run the lifespan, but this confirms imports are clean.)

- [ ] **Step 4: Commit**

```bash
git add expert_call_agent/api/app.py
git commit -m "feat: initialize Weave on app startup

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Trace the client layer (the lowest-level Vertex/Cloud calls)

**Files:**
- Modify: `expert_call_agent/clients/claude_client.py`
- Modify: `expert_call_agent/clients/gemini_client.py`
- Modify: `expert_call_agent/clients/cloud_stt_client.py`
- Modify: `expert_call_agent/clients/stt_client.py`

For each file: add `from observability import op` alongside the existing top-of-file imports, then put `@op()` on its own line directly above the target `async def`, at the method's indentation (4 spaces).

- [ ] **Step 1: `claude_client.py` — decorate `generate`**

Add the import at the top, then above `async def generate(` (line 21):

```python
    @op()
    async def generate(
```

- [ ] **Step 2: `gemini_client.py` — decorate `generate` and `generate_with_audio`**

Add the import at the top, then `@op()` above `async def generate(` (line 18) **and** above `async def generate_with_audio(` (line 42):

```python
    @op()
    async def generate(
```
```python
    @op()
    async def generate_with_audio(
```

- [ ] **Step 3: `cloud_stt_client.py` — decorate `transcribe` (Chirp 2, the default STT)**

Add the import at the top, then above `async def transcribe(` (line 23):

```python
    @op()
    async def transcribe(
```

- [ ] **Step 4: `stt_client.py` — decorate `transcribe` (Gemini STT fallback)**

`stt_client.py` already starts with `import config` (line 1). Add `from observability import op` after it, then above `async def transcribe(` (line 9):

```python
    @op()
    async def transcribe(
```

- [ ] **Step 5: Import smoke**

Run (from `expert_call_agent/`):
```bash
WEAVE_DISABLED=1 python3 -c "
from clients.claude_client import ClaudeClient
from clients.gemini_client import GeminiClient
from clients.stt_client import STTClient
from clients.cloud_stt_client import CloudSTTClient
print('clients import OK')
"
```
Expected: prints `clients import OK`.

- [ ] **Step 6: Commit**

```bash
git add expert_call_agent/clients/claude_client.py expert_call_agent/clients/gemini_client.py expert_call_agent/clients/cloud_stt_client.py expert_call_agent/clients/stt_client.py
git commit -m "feat: trace client LLM/STT calls with Weave

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Trace the agent layer (builds the per-agent trace trees)

Decorating `_call_model` + `_call_model_text` once in the base class, plus each agent's `run`, yields nested traces like `qa_agent.run → BaseAgent._call_model → ClaudeClient.generate` and `orchestrator.run → BaseAgent._call_model_text → ClaudeClient.generate`.

**Files:**
- Modify: `expert_call_agent/agents/base_agent.py`
- Modify: `expert_call_agent/agents/qa_agent.py`
- Modify: `expert_call_agent/agents/followup_agent.py`
- Modify: `expert_call_agent/agents/note_taker.py`
- Modify: `expert_call_agent/agents/orchestrator.py`
- Modify: `expert_call_agent/agents/context_ingestion.py`
- Modify: `expert_call_agent/agents/call_guide_drafter.py`
- Modify: `expert_call_agent/agents/post_call_summarizer.py`

- [ ] **Step 1: `base_agent.py` — decorate both model-call helpers**

Add `from observability import op` after `from models import CallSession` (line 8). Then add `@op()` directly above **both**:

```python
    @op()
    async def _call_model(self, user_prompt: str) -> str:
```
```python
    @op()
    async def _call_model_text(self, user_prompt: str) -> str:
```

> Do **not** decorate `_parse_json` / `_safe_parse_json` (pure, non-LLM helpers).

- [ ] **Step 2: Decorate every agent's `run`**

In **each** of these seven files, add `from observability import op` alongside the existing imports at the top, then put `@op()` on the line directly above its `async def run(`. The exact `run` lines:

- `qa_agent.py` (line 27): `    async def run(self, session: CallSession, **kwargs) -> AgentAction | None:`
- `followup_agent.py` (line 35): `    async def run(self, session: CallSession, **kwargs) -> list[AgentAction]:`
- `note_taker.py` (line 36): `    async def run(self, session: CallSession, **kwargs) -> list[AgentAction]:`
- `orchestrator.py` (line 24): `    async def run(self, session: CallSession, flag: AgentAction | None = None, **kwargs) -> str:`
- `context_ingestion.py` (line 32): `    async def run(self, session: CallSession, brief_text: str = "") -> ProjectContext:`
- `call_guide_drafter.py` (line 32): `    async def run(self, session: CallSession, **kwargs) -> CallGuide:`
- `post_call_summarizer.py` (line 23): `    async def run(self, session: CallSession, **kwargs) -> dict:`

Each becomes, e.g.:

```python
    @op()
    async def run(self, session: CallSession, **kwargs) -> AgentAction | None:
```

> Do **not** decorate the agents' `get_latest_notes` / `get_latest_coverage` / `seed_processed` helpers — only `run`.

- [ ] **Step 3: Import smoke (all agents construct + expose `run`)**

Run (from `expert_call_agent/`):
```bash
WEAVE_DISABLED=1 python3 -c "
from agents.qa_agent import QAAgent
from agents.followup_agent import FollowUpAgent
from agents.note_taker import NoteTakerAgent
from agents.orchestrator import OrchestratorAgent
from agents.context_ingestion import ContextIngestionAgent
from agents.call_guide_drafter import CallGuideDrafterAgent
from agents.post_call_summarizer import PostCallSummarizerAgent
for c in (QAAgent, FollowUpAgent, NoteTakerAgent, OrchestratorAgent, ContextIngestionAgent, CallGuideDrafterAgent, PostCallSummarizerAgent):
    assert callable(getattr(c, 'run')), c
print('agents import OK')
"
```
Expected: prints `agents import OK`.

> If a class name above differs from the actual class name in its file, use the real class name (open the file's `class …(BaseAgent)` line). The import paths (module names) are correct.

- [ ] **Step 4: Commit**

```bash
git add expert_call_agent/agents/
git commit -m "feat: trace agent.run + _call_model/_call_model_text with Weave

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Eval scorers (pure functions)

**Files:**
- Create: `expert_call_agent/evals/__init__.py`
- Create: `expert_call_agent/evals/scorers.py`

- [ ] **Step 1: Create the package marker**

Create `expert_call_agent/evals/__init__.py` (empty file).

- [ ] **Step 2: Create the scorers**

Create `expert_call_agent/evals/scorers.py` with exactly:

```python
"""Pure scoring functions for Weave evaluations.

Each scorer takes dataset columns (by name) plus the model `output` dict and
returns a dict of metrics. No I/O, so they are trivially checkable.
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

> Note on `None`: when the QA agent stays silent it returns `None`, and the runner (Task 8) maps that to `{}` before scoring, so `flag_type_match` records `correct=False` and `produced_question` records `nonempty=False` rather than crashing.

- [ ] **Step 3: Inline smoke (no pytest)**

Run (from `expert_call_agent/`):
```bash
python3 -c "
from evals.scorers import flag_type_match, produced_question
assert flag_type_match('must_ask', {'flag_type': 'must_ask'}) == {'correct': True}
assert flag_type_match('must_ask', {'flag_type': 'nice_to_have'}) == {'correct': False}
assert flag_type_match('must_ask', {}) == {'correct': False}
assert flag_type_match('must_ask', None) == {'correct': False}
assert produced_question({'content': 'What is the GPU count?'}) == {'nonempty': True}
assert produced_question({'content': '   '}) == {'nonempty': False}
assert produced_question({}) == {'nonempty': False}
print('scorers OK')
"
```
Expected: prints `scorers OK`.

- [ ] **Step 4: Commit**

```bash
git add expert_call_agent/evals/__init__.py expert_call_agent/evals/scorers.py
git commit -m "feat: add Weave eval scorers (flag-type match, produced-question)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Eval dataset

**Files:**
- Create: `expert_call_agent/evals/flag_dataset.json`

Each row is one expert utterance plus the flag tier a good QA agent should produce next. Rows are seeded from the CoreWeave demo in `api/routes_call.py` (`DEMO_EXPERT_RESPONSES`), so the eval matches the demo's domain.

- [ ] **Step 1: Create the dataset**

Create `expert_call_agent/evals/flag_dataset.json` with exactly:

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

> Labels are a starting hypothesis (hard numbers/contradictions → `must_ask`; context/color → `should_ask`). The eval measures how often the agent agrees; adjust labels as the team aligns on "good."

- [ ] **Step 2: Validate the JSON parses**

Run (from `expert_call_agent/`):
```bash
python3 -c "import json; rows=json.load(open('evals/flag_dataset.json')); print(len(rows), 'rows'); assert all('transcript' in r and 'expected_flag' in r for r in rows)"
```
Expected: prints `5 rows`, no assertion error.

- [ ] **Step 3: Commit**

```bash
git add expert_call_agent/evals/flag_dataset.json
git commit -m "feat: add QA flag-quality eval dataset (CoreWeave-seeded)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Eval runner (`weave.Evaluation`)

**Files:**
- Create: `expert_call_agent/evals/run_flag_eval.py`

> Requires `WANDB_API_KEY` set and working Vertex ADC — it makes real Claude (QA agent) calls. QA uses Claude by default (`config.AGENT_MODELS`).

- [ ] **Step 1: Write the runner**

Create `expert_call_agent/evals/run_flag_eval.py` with exactly:

```python
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
```

- [ ] **Step 2: Parse / import check (no network)**

Run (from `expert_call_agent/`):
```bash
WEAVE_DISABLED=1 python3 -c "import ast; ast.parse(open('evals/run_flag_eval.py').read()); print('parses OK')"
```
Expected: prints `parses OK`.

- [ ] **Step 3: Commit**

```bash
git add expert_call_agent/evals/run_flag_eval.py
git commit -m "feat: add Weave evaluation runner for QA flag quality

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

(The real run happens in Task 10, Step 3, once a key is set.)

> **Version note:** if `evaluation.evaluate(predict)` rejects a bare op in the installed weave version, wrap it in a `weave.Model` subclass with an `@weave.op() async def predict(self, transcript)` and pass the instance. See https://weave-docs.wandb.ai/guides/core-types/evaluations.

---

## Task 9: Document env vars (`.env.example`)

**Files:**
- Create: `expert_call_agent/.env.example`

- [ ] **Step 1: Create `.env.example`**

Create `expert_call_agent/.env.example` with exactly:

```bash
# Weave / W&B observability
# Get a key at https://wandb.ai/authorize
WANDB_API_KEY=
# Weave project name (optional; defaults to http418-expert-call)
WEAVE_PROJECT=http418-expert-call
# Set to 1 to fully disable Weave tracing (app still runs). Use this if you have
# no WANDB_API_KEY, so weave.op degrades to a pure no-op.
WEAVE_DISABLED=
```

- [ ] **Step 2: Confirm `.env` is gitignored, `.env.example` is not**

Run (from repo root `/home/kevin/EN-automation`):
```bash
git check-ignore expert_call_agent/.env && echo ".env ignored (good)"
git check-ignore expert_call_agent/.env.example && echo "WARN: .env.example IS ignored (should NOT be)" || echo ".env.example tracked (good)"
```
Expected: `.env ignored (good)` then `.env.example tracked (good)`. If `.env` is **not** ignored, add `.env` to `.gitignore` and commit that too.

- [ ] **Step 3: Commit**

```bash
git add expert_call_agent/.env.example
git commit -m "docs: document Weave env vars in .env.example

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: End-to-end verification (disabled smoke, then real run with key)

**Files:** none (verification only).

- [ ] **Step 1: Full no-op smoke (no key needed)**

Run (from `expert_call_agent/`):
```bash
WEAVE_DISABLED=1 python3 -c "
from api.app import create_app; create_app()
from evals.scorers import flag_type_match, produced_question
assert produced_question({'content':'x'})['nonempty'] is True
print('disabled smoke OK')
"
```
Expected: prints `disabled smoke OK` — proves the whole app + evals import and run with weave fully off.

- [ ] **Step 2: Real tracing — run the demo and inspect the trace tree**

Set the key, launch the server on **8899**, and drive the demo over the IAP tunnel:
```bash
cd /home/kevin/EN-automation/expert_call_agent
lsof -ti :8899 2>/dev/null | xargs -r kill -9 2>/dev/null
WANDB_API_KEY=<your-key> WEAVE_PROJECT=http418-expert-call nohup python3 -c "
import uvicorn
from api.app import create_app
uvicorn.run(create_app(), host='0.0.0.0', port=8899)
" > /tmp/weave_smoke.log 2>&1 &
sleep 4
grep -i "weave\|View at\|wandb" /tmp/weave_smoke.log | head
```
Then in the browser over the tunnel (`http://localhost:8899/`): **Use Sample Brief → Generate Call Guide → Run Demo Simulation**.

Expected: the Weave URL is printed in `/tmp/weave_smoke.log` on startup; the Weave UI then shows, per expert turn, sibling trees `qa_agent.run`, `followup_agent.run`, `note_taker.run` (each nesting `_call_model → ClaudeClient.generate`) plus an `orchestrator.run → _call_model_text → ClaudeClient.generate` for the spoken turn, and `call_guide_drafter.run` / `context_ingestion.run` from the pre-call step. Stop the server when done (`lsof -ti :8899 | xargs -r kill -9`).

- [ ] **Step 3: Real eval — run the evaluation**

Run (from `expert_call_agent/`):
```bash
WANDB_API_KEY=<your-key> python3 -m evals.run_flag_eval
```
Expected: Weave prints a project URL and an evaluation summary with `flag_type_match.correct` and `produced_question.nonempty` means; the UI shows a `predict` trace per dataset row, each nesting `qa_agent.run → _call_model → ClaudeClient.generate`.

- [ ] **Step 4: Capture demo assets + final commit (if any fixes were needed)**

Screenshot the trace tree and the eval table for the pitch. If Steps 1–3 surfaced fixes:
```bash
git add -A
git commit -m "fix: address Weave integration smoke findings

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
If no fixes were needed, skip the commit.

---

## Done / demo checklist

- [ ] `WEAVE_DISABLED=1 python3 -c "from api.app import create_app; create_app()"` → builds with no W&B account.
- [ ] `WANDB_API_KEY=… python3 …:8899` + Run Demo → Weave shows nested per-agent trace trees.
- [ ] `WANDB_API_KEY=… python3 -m evals.run_flag_eval` → Weave shows an eval with flag-quality scores.
- [ ] Trace-tree + eval-table screenshots saved for the pitch.

## How this maps to judging

- **Agent Orchestration / Technical Execution:** the Weave trees literally show the parallel `qa/followup/note_taker` agents and the orchestrator's spoken turn, each drilling into the exact Vertex call — proof, not claims.
- **Best Use of Weave:** tracing *and* an evaluation harness with custom scorers (not just autologging).
- **Sponsor Usage:** Weave (W&B) used end-to-end alongside Claude/Gemini on Vertex.

## Future evals (out of scope here — documented so they aren't lost)

- **Follow-up contradiction recall:** dataset of expert utterances that conflict with `project_context.known_data_points`; scorer checks the Follow-up agent emitted a `contradiction` flag. (Directly measures the suppression work.)
- **Note factuality (LLM-as-judge):** compare `note_taker` output against the source utterance.
- **Per-turn unified trace:** if the "agents/clients only" trace ever feels too fragmented for the demo, add one `@op` root around the `process_entry` fan-out in `routes_call.py` so each expert turn is a single tree (deferred to avoid touching the live hot path).
