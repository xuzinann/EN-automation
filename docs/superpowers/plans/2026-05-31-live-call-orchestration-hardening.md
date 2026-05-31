# Live-Call Orchestration Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the live-call backend (`expert_call_agent/`) for availability, robustness, and conversational quality — without changing the WebSocket message contract the frontend depends on.

**Architecture:** The live loop stays as-is in shape — per expert turn, three flagger agents (QA, Follow-up, Note-taker) run in parallel, a deterministic `LiveCallQueue` selects one flag, the Orchestrator phrases it, and it is auto-spoken via TTS. This plan adds: per-call LLM timeouts, cached/non-blocking GCP auth, a single-connection guard, graceful JSON degradation, a conversational cooldown + TTL + near-duplicate dedup in the queue, low sampling temperature for the structured agents, note-taker speaker filtering, and a dead-code sweep.

**Tech Stack:** Python 3, FastAPI + WebSockets, Pydantic v2, google-genai (Vertex), Claude via Vertex `rawPredict`, Google Cloud TTS REST. No new dependencies.

---

## Conventions for this repo (READ FIRST)

- **No feature branch.** Commit straight to `main` (per project convention). Each task ends with a commit.
- **No committed test suite.** This is a hackathon repo — do **not** add `pytest` files. Verification is lightweight: `py_compile`, a throwaway in-process queue check, and an optional end-to-end smoke. The verification scripts in Task 9 are run ad hoc and **not** committed.
- **Run Python as `python3`** (there is no `python` on PATH on this VM).
- **Module imports are top-level** (`import config`, `from clients.gemini_client import ...`). The app inserts the package root onto `sys.path` in `api/app.py`. Run ad-hoc scripts from inside `expert_call_agent/` so these imports resolve.
- **Do NOT change the WebSocket message contract.** Server→client types stay: `ai_turn` {question, agent, flag_type, rationale}, `tts_audio` {data}, `transcript`, `note`, `coverage_update`, `demo_progress`, `demo_complete`, `error`. Client→server: `text_input`, `run_demo`, raw audio bytes.

## What this plan fixes (traceability)

| # | Issue | Task |
|---|-------|------|
| 1 | No per-call LLM timeout → one hung call stalls the whole WS loop | T1, T3, T8 |
| 2 | Blocking, per-request GCP auth on the event loop | T2 |
| 3 | Reconnect duplicates notes / re-asks questions | T6, T7, T8 |
| 4 | AI speaks every turn (QA always emits; QA tiers uncapped) | T1, T5, T6 |
| 5 | Exact-match-only dedup; rephrasings slip through | T1, T6 |
| 6 | JSON parse is fatal per turn; QA-on-Claude most fragile | T4, T5 |
| 7 | `gemini.generate` can return `None` → AttributeError | T4, T5 |
| 8 | Unbounded, ageless pending queue (stale flags resurface) | T1, T6 |
| 9 | Latency: QA on Claude Sonnet on the critical path | Deferred (see Out of scope) — one-line config lever |
| 10 | Note-taker ingests the AI's own turns | T5 |
| Low | Dead code (`add_action`, `pop_pending_actions`, `pending_actions`, `update_takeaways`, `ORCHESTRATOR_TICK_SECONDS`, `AgentAction.priority`) | T1, T5, T7 |
| Low | Latent stale-notes re-emit; redundant `get_session`; no temperature | T4, T5, T8 |

## Explicitly OUT of scope (intentional, with rationale)

- **Session / lock eviction.** In-memory demo; auto-evicting risks killing an active session. Deferred.
- **`run_demo` blocking the receive loop.** The single-connection guard (T7/T8) removes the concurrency risk; a demo button blocking for ~10s is acceptable.
- **Full queue-state persistence across reconnect.** On reconnect we prevent duplicate *notes* (T6/T8) and prevent a *parallel* loop (T7/T8); re-asking a question after a mid-call reconnect is acceptable for the demo. Persisting `_asked` to the session model is deferred.
- **Embeddings-based semantic dedup.** Token-set Jaccard (T6) is the dependency-free approximation; embeddings are overkill here.
- **QA latency (#9).** QA stays on Claude Sonnet for question quality. It's the slowest model on the live critical path, but `_safe_parse_json` (T4) makes the no-JSON-mode path degrade gracefully. If latency becomes a problem, switch the `qa_agent` line in `config.AGENT_MODELS` to `("gemini", GEMINI_FLASH_MODEL)` — a one-line change, no other edits.

---

## Task 1: Config — timeouts, caps, cooldown, TTL, dedup, QA model move, dead-config removal

**Files:**
- Modify: `expert_call_agent/config.py` (full replace)

- [ ] **Step 1: Replace `config.py` with the version below**

Changes vs current: removes the unused `ORCHESTRATOR_TICK_SECONDS`; adds timeout/temperature knobs and the new live-call policy knobs (caps now also bound `should_ask`/`nice_to_have` — issue #4). `qa_agent` stays on Claude Sonnet (issue #9 deferred); the comment on that line documents the one-line switch to Gemini Flash if latency becomes a problem.

```python
GCP_PROJECT = "gp-ct-sbox-sat-gcp0bg-darksoft"

GEMINI_REGION = "us-central1"
CLAUDE_REGION = "us-east5"

GEMINI_PRO_MODEL = "gemini-2.5-pro"
GEMINI_FLASH_MODEL = "gemini-2.5-flash"
CLAUDE_MODEL = "claude-sonnet-4@20250514"

ANTHROPIC_VERSION = "vertex-2023-10-16"

TTS_ENDPOINT = "https://texttospeech.googleapis.com/v1/text:synthesize"
TTS_VOICE = "en-US-Studio-O"
TTS_LANGUAGE = "en-US"

AGENT_MODELS = {
    "context_ingestion": ("claude", CLAUDE_MODEL),
    "call_guide_drafter": ("claude", CLAUDE_MODEL),
    "orchestrator": ("gemini", GEMINI_FLASH_MODEL),
    # Kept on Claude Sonnet for question quality. This line is the single lever
    # for the live-call latency tradeoff (#9): switching to
    # ("gemini", GEMINI_FLASH_MODEL) is a one-line change with no other code
    # edits — BaseAgent._call_model branches on provider, and _safe_parse_json
    # makes the no-JSON-mode path degrade to "stay silent" instead of crashing.
    "qa_agent": ("claude", CLAUDE_MODEL),
    "followup_agent": ("gemini", GEMINI_PRO_MODEL),
    "note_taker": ("gemini", GEMINI_FLASH_MODEL),
    "post_call_summarizer": ("claude", CLAUDE_MODEL),
}

MAX_TRANSCRIPT_CONTEXT = 20

# Hard ceiling on any single live-call agent call, so one hung LLM request
# can't stall the whole WebSocket loop. Not an SLA — just a stall guard.
AGENT_TIMEOUT_SECONDS = 20.0

# HTTP timeout (milliseconds) for the google-genai SDK, so the underlying
# worker thread can't block indefinitely behind asyncio.wait_for cancellation.
GEMINI_HTTP_TIMEOUT_MS = 30000

# Sampling temperature for the structured (JSON-emitting) live agents. Low =
# stabler turn-to-turn output and more reliable parsing.
STRUCTURED_TEMPERATURE = 0.2

# Live-call flag priority: lower index = higher priority. FIFO breaks ties.
FLAG_TIER_ORDER = ["contradiction", "must_ask", "probe", "should_ask", "nice_to_have"]

# Max times each flag type may be surfaced in a single live call (absent = unlimited).
# Caps now also bound QA's tiers so the AI doesn't ask endlessly.
LIVE_FLAG_CAPS = {"contradiction": 5, "probe": 5, "should_ask": 8, "nice_to_have": 4}

# A pending flag not surfaced within this many expert turns is dropped as stale.
LIVE_FLAG_TTL_TURNS = 4

# Conversational cooldown: after the AI speaks it stays quiet for this many
# expert turns unless an urgent flag arrives — so it doesn't interject after
# every single sentence.
LIVE_MIN_TURNS_BETWEEN_QUESTIONS = 2
LIVE_COOLDOWN_EXEMPT_TIERS = {"contradiction", "must_ask"}

# Near-duplicate suppression: a candidate flag whose token-set Jaccard overlap
# with an already-asked or still-pending flag is >= this is treated as a duplicate.
LIVE_DEDUP_JACCARD = 0.8
```

- [ ] **Step 2: Verify it imports**

Run: `cd expert_call_agent && python3 -c "import config; print(config.AGENT_MODELS['qa_agent'], config.AGENT_TIMEOUT_SECONDS, config.LIVE_FLAG_CAPS)"`
Expected: `('claude', 'claude-sonnet-4@20250514') 20.0 {'contradiction': 5, 'probe': 5, 'should_ask': 8, 'nice_to_have': 4}`

- [ ] **Step 3: Commit**

```bash
git add expert_call_agent/config.py
git commit -m "config: add live-call timeout/cooldown/TTL/dedup knobs"
```

---

## Task 2: Cached, non-blocking GCP auth (issue #2)

**Files:**
- Create: `expert_call_agent/clients/gcp_auth.py`
- Modify: `expert_call_agent/clients/claude_client.py` (full replace)
- Modify: `expert_call_agent/clients/tts_client.py` (full replace)

`google.auth.default()` and `Credentials.refresh()` do blocking network/disk I/O and currently run on the event loop on **every** Claude and TTS request. This task resolves ADC once, caches it, refreshes only when the token is expired, and offloads both operations to a thread.

- [ ] **Step 1: Create `clients/gcp_auth.py`**

```python
"""Cached, non-blocking Application Default Credentials access.

google.auth.default() and Credentials.refresh() do blocking network/disk I/O,
so they must not run on the event loop. This module resolves ADC once, caches
the credentials, refreshes only when the token has expired, and offloads both
operations to a worker thread.
"""
import asyncio

import google.auth
import google.auth.transport.requests

_credentials = None
_project: str | None = None
_lock = asyncio.Lock()


async def get_auth() -> tuple[str, str]:
    """Return (bearer_token, project), refreshing the token only when expired."""
    global _credentials, _project
    async with _lock:
        if _credentials is None:
            _credentials, _project = await asyncio.to_thread(google.auth.default)
        if not _credentials.valid:
            await asyncio.to_thread(
                _credentials.refresh,
                google.auth.transport.requests.Request(),
            )
        return _credentials.token, (_project or "")
```

- [ ] **Step 2: Replace `clients/claude_client.py`**

```python
import httpx

import config
from clients.gcp_auth import get_auth


class ClaudeClient:
    def __init__(self):
        self.project = config.GCP_PROJECT
        self.region = config.CLAUDE_REGION
        self.model = config.CLAUDE_MODEL
        self._http_client = httpx.AsyncClient(timeout=60.0)

    async def _get_headers(self) -> dict:
        token, _ = await get_auth()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    async def generate(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> str:
        url = (
            f"https://{self.region}-aiplatform.googleapis.com/v1/"
            f"projects/{self.project}/locations/{self.region}/"
            f"publishers/anthropic/models/{self.model}:rawPredict"
        )
        body = {
            "anthropic_version": config.ANTHROPIC_VERSION,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if system:
            body["system"] = system
        if temperature is not None:
            body["temperature"] = temperature

        headers = await self._get_headers()
        resp = await self._http_client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]

    async def close(self):
        await self._http_client.aclose()
```

- [ ] **Step 3: Replace `clients/tts_client.py`**

```python
import base64

import httpx

import config
from clients.gcp_auth import get_auth


class TTSClient:
    def __init__(self):
        self._http_client = httpx.AsyncClient(timeout=30.0)

    async def synthesize(self, text: str, voice: str | None = None) -> bytes:
        token, project = await get_auth()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "x-goog-user-project": project,
        }
        body = {
            "input": {"text": text},
            "voice": {
                "languageCode": config.TTS_LANGUAGE,
                "name": voice or config.TTS_VOICE,
            },
            "audioConfig": {"audioEncoding": "MP3"},
        }
        resp = await self._http_client.post(
            config.TTS_ENDPOINT, headers=headers, json=body
        )
        resp.raise_for_status()
        return base64.b64decode(resp.json()["audioContent"])

    async def close(self):
        await self._http_client.aclose()
```

- [ ] **Step 4: Verify imports**

Run: `cd expert_call_agent && python3 -c "from clients.gcp_auth import get_auth; from clients.claude_client import ClaudeClient; from clients.tts_client import TTSClient; print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add expert_call_agent/clients/gcp_auth.py expert_call_agent/clients/claude_client.py expert_call_agent/clients/tts_client.py
git commit -m "clients: cache ADC + offload auth refresh off the event loop; add temperature param to Claude"
```

---

## Task 3: Gemini client — HTTP timeout + temperature (issues #1, #7 support)

**Files:**
- Modify: `expert_call_agent/clients/gemini_client.py` (full replace)

Adds an SDK-level HTTP timeout (so a hung call can't block a worker thread forever even after `asyncio.wait_for` cancels the awaitable) and a `temperature` passthrough. `HttpOptions(timeout=<ms>)` is confirmed available in google-genai 2.7.0.

- [ ] **Step 1: Replace `clients/gemini_client.py`**

```python
import asyncio

from google import genai
from google.genai import types

import config


class GeminiClient:
    def __init__(self):
        self.client = genai.Client(
            vertexai=True,
            project=config.GCP_PROJECT,
            location=config.GEMINI_REGION,
            http_options=types.HttpOptions(timeout=config.GEMINI_HTTP_TIMEOUT_MS),
        )

    async def generate(
        self,
        model: str,
        prompt: str,
        system_instruction: str | None = None,
        response_mime_type: str | None = None,
        temperature: float | None = None,
    ) -> str:
        gen_config = types.GenerateContentConfig()
        if system_instruction:
            gen_config.system_instruction = system_instruction
        if response_mime_type:
            gen_config.response_mime_type = response_mime_type
        if temperature is not None:
            gen_config.temperature = temperature

        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=model,
            contents=prompt,
            config=gen_config,
        )
        return response.text

    async def generate_with_audio(
        self,
        model: str,
        audio_bytes: bytes,
        mime_type: str,
        prompt: str,
    ) -> str:
        audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
        text_part = types.Part.from_text(text=prompt)

        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=model,
            contents=[audio_part, text_part],
        )
        return response.text
```

- [ ] **Step 2: Verify the SDK accepts the client options**

Run: `cd expert_call_agent && python3 -c "from clients.gemini_client import GeminiClient; GeminiClient(); print('client constructed ok')"`
Expected: `client constructed ok`
**If this raises** a `TypeError`/validation error on `http_options=` (SDK version drift), remove the `http_options=...` line from `__init__` and rely on the `asyncio.wait_for` guard in Task 8 alone; note the removal in the commit message.

- [ ] **Step 3: Commit**

```bash
git add expert_call_agent/clients/gemini_client.py
git commit -m "clients: add Gemini HTTP timeout + temperature passthrough"
```

---

## Task 4: BaseAgent — safe JSON parsing + temperature plumbing (issues #6, #7)

**Files:**
- Modify: `expert_call_agent/agents/base_agent.py` (full replace)

Adds `_safe_parse_json` (returns a default on empty/None/malformed output instead of raising) and a `temperature` class attribute plumbed into both providers.

- [ ] **Step 1: Replace `agents/base_agent.py`**

```python
import json
import re
from abc import ABC, abstractmethod

import config
from clients.gemini_client import GeminiClient
from clients.claude_client import ClaudeClient
from models import CallSession


class BaseAgent(ABC):
    name: str
    system_prompt: str
    temperature: float | None = None  # subclasses override; None = provider default

    def __init__(self, gemini_client: GeminiClient, claude_client: ClaudeClient):
        self.gemini = gemini_client
        self.claude = claude_client
        provider, model = config.AGENT_MODELS[self.name]
        self.model_provider = provider
        self.model_name = model

    async def _call_model(self, user_prompt: str) -> str:
        if self.model_provider == "gemini":
            return await self.gemini.generate(
                model=self.model_name,
                prompt=user_prompt,
                system_instruction=self.system_prompt,
                response_mime_type="application/json",
                temperature=self.temperature,
            )
        else:
            return await self.claude.generate(
                messages=[{"role": "user", "content": user_prompt}],
                system=self.system_prompt,
                temperature=self.temperature,
            )

    def _parse_json(self, text: str) -> dict | list:
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        return json.loads(cleaned)

    def _safe_parse_json(self, text: str | None, default):
        """Parse model JSON, returning `default` on empty/None/malformed output.

        Live-call agents must degrade gracefully: a single bad turn should fall
        back to a safe default, never raise and stall the call.
        """
        if not text:
            return default
        try:
            return self._parse_json(text)
        except ValueError:  # json.JSONDecodeError is a subclass of ValueError
            return default

    @abstractmethod
    async def run(self, session: CallSession, **kwargs):
        ...
```

- [ ] **Step 2: Verify import**

Run: `cd expert_call_agent && python3 -c "from agents.base_agent import BaseAgent; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add expert_call_agent/agents/base_agent.py
git commit -m "agents: add safe JSON parsing + temperature plumbing to BaseAgent"
```

---

## Task 5: Live agents — QA/Follow-up/Note-taker/Orchestrator (issues #4, #5, #6, #7, #10, dead code)

**Files:**
- Modify: `expert_call_agent/agents/qa_agent.py` (full replace)
- Modify: `expert_call_agent/agents/followup_agent.py` (full replace)
- Modify: `expert_call_agent/agents/note_taker.py` (full replace)
- Modify: `expert_call_agent/agents/orchestrator.py` (full replace)

Across all four: drop the now-unused `priority=` kwarg (the `AgentAction.priority` field is removed in Task 7), set low `temperature`, and use safe parsing. QA can now return `None` (stay silent). Note-taker filters to `expert` turns (so it never extracts from the AI's own questions) and resets `_latest_notes` on the empty path. Orchestrator guards against a `None` response.

- [ ] **Step 1: Replace `agents/qa_agent.py`**

```python
from agents.base_agent import BaseAgent
from models import CallSession, AgentAction
import config

_VALID_FLAGS = {"must_ask", "should_ask", "nice_to_have"}


class QAAgent(BaseAgent):
    name = "qa_agent"
    temperature = config.STRUCTURED_TEMPERATURE
    system_prompt = (
        "You are the lead interviewer in a PE expert call. "
        "Given the interview guide and real-time transcript, suggest the best next "
        "question to ask. Adapt based on what the expert has said — don't repeat "
        "questions already answered, and probe deeper when answers are vague.\n\n"
        "If the expert's last answer is still unfolding and no new question is "
        "warranted yet, return an empty question string to stay silent.\n\n"
        "Return ONLY valid JSON:\n"
        '{"question": "the question to ask, or empty string to stay silent", '
        '"rationale": "why this question now", '
        '"guide_section": "which section this addresses", '
        '"priority": "must_ask|should_ask|nice_to_have"}\n'
        "Set priority from the guide question's own priority (default should_ask).\n"
        "No markdown fences or commentary."
    )

    async def run(self, session: CallSession, **kwargs) -> AgentAction | None:
        guide_json = session.call_guide.model_dump_json() if session.call_guide else "{}"
        recent = session.transcript[-config.MAX_TRANSCRIPT_CONTEXT:]
        transcript = "\n".join(f"[{e.speaker}] {e.text}" for e in recent)

        prompt = (
            f"Interview guide:\n{guide_json}\n\n"
            f"Transcript so far:\n{transcript}\n\n"
            "Suggest the single best next question to ask the expert, or an empty "
            "question if it's better to stay silent for now."
        )
        response = await self._call_model(prompt)
        parsed = self._safe_parse_json(response, {})

        question = (parsed.get("question") or "").strip()
        if not question:
            return None

        flag_type = parsed.get("priority", "should_ask")
        if flag_type not in _VALID_FLAGS:
            flag_type = "should_ask"

        return AgentAction(
            agent_name=self.name,
            action_type="suggest_question",
            flag_type=flag_type,
            content=question,
            metadata={
                "rationale": parsed.get("rationale", ""),
                "section": parsed.get("guide_section", ""),
            },
        )
```

- [ ] **Step 2: Replace `agents/followup_agent.py`**

```python
from agents.base_agent import BaseAgent
from models import CallSession, AgentAction, CoverageStatus
import config


class FollowUpAgent(BaseAgent):
    name = "followup_agent"
    temperature = config.STRUCTURED_TEMPERATURE
    system_prompt = (
        "You monitor an expert call for coverage completeness. "
        "Track which interview guide sections have been addressed. "
        "Flag when the expert gives vague or incomplete answers needing follow-up. "
        "Detect contradictions with known data points.\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "coverage": [{"section": "string", "covered_pct": 0.0-1.0, '
        '"answered_questions": ["strings"], "remaining_questions": ["strings"]}],\n'
        '  "followups": [{"question": "string", "reason": "string"}],\n'
        '  "contradictions": [{"claim": "string", "conflicts_with": "string"}]\n'
        "}\n"
        "No markdown fences or commentary."
    )

    async def run(self, session: CallSession, **kwargs) -> list[AgentAction]:
        guide_json = session.call_guide.model_dump_json() if session.call_guide else "{}"
        context_json = session.project_context.model_dump_json() if session.project_context else "{}"
        recent = session.transcript[-config.MAX_TRANSCRIPT_CONTEXT:]
        transcript = "\n".join(f"[{e.speaker}] {e.text}" for e in recent)

        prompt = (
            f"Interview guide:\n{guide_json}\n\n"
            f"Known data points:\n{context_json}\n\n"
            f"Transcript so far:\n{transcript}\n\n"
            "Analyze coverage completeness, suggest follow-ups, and flag contradictions."
        )
        response = await self._call_model(prompt)
        parsed = self._safe_parse_json(response, {})

        actions = []
        for fu in parsed.get("followups", []):
            question = (fu.get("question") or "").strip()
            if not question:
                continue
            actions.append(AgentAction(
                agent_name=self.name,
                action_type="followup",
                flag_type="probe",
                content=question,
                metadata={"reason": fu.get("reason", "")},
            ))

        for contradiction in parsed.get("contradictions", []):
            actions.append(AgentAction(
                agent_name=self.name,
                action_type="contradiction",
                flag_type="contradiction",
                content=(
                    f"There may be an inconsistency: "
                    f"{contradiction.get('claim', '')} vs "
                    f"{contradiction.get('conflicts_with', '')}. Could you clarify?"
                ),
                metadata={"reason": "contradiction with known data"},
            ))

        self._latest_coverage = [
            CoverageStatus.model_validate(c)
            for c in parsed.get("coverage", [])
        ]

        return actions

    def get_latest_coverage(self) -> list[CoverageStatus]:
        return getattr(self, "_latest_coverage", [])
```

- [ ] **Step 3: Replace `agents/note_taker.py`**

Key changes: `seed_processed()` for reconnect (Task 8 uses it); always advance `_last_processed`; filter to `expert` turns before extracting; reset `_latest_notes` on the empty path (fixes latent stale-notes re-emit); per-note validation guard; safe parse; low temperature; no `priority=` kwarg.

```python
from agents.base_agent import BaseAgent
from models import CallSession, AgentAction, StructuredNote
import config


class NoteTakerAgent(BaseAgent):
    name = "note_taker"
    temperature = config.STRUCTURED_TEMPERATURE
    system_prompt = (
        "Extract structured notes from an expert call transcript. "
        "Categorize each note:\n"
        "- headcount: N-sizing data by role\n"
        "- org_structure: reporting lines, departments\n"
        "- market: competitive positioning, market share\n"
        "- operations: processes, technology, efficiency\n"
        "- financials: revenue, costs, margins\n"
        "- quote: verbatim quote worth preserving\n"
        "- other: anything else noteworthy\n\n"
        "Assess confidence: stated (expert said directly), estimated (implied), "
        "inferred (you derived).\n\n"
        "Return ONLY valid JSON array:\n"
        '[{"category": "string", "content": "string", "confidence": "stated|estimated|inferred", '
        '"source_quote": "string or null"}]\n'
        "No markdown fences or commentary. Return [] if no new notes."
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_processed = 0
        self._latest_notes: list[StructuredNote] = []

    def seed_processed(self, count: int) -> None:
        """Skip note extraction for entries already in the transcript (reconnect)."""
        self._last_processed = count

    async def run(self, session: CallSession, **kwargs) -> list[AgentAction]:
        new_entries = session.transcript[self._last_processed:]
        self._last_processed = len(session.transcript)

        # Only extract facts from the expert — never from the AI's own questions.
        expert_entries = [e for e in new_entries if e.speaker == "expert"]
        if not expert_entries:
            self._latest_notes = []
            return []

        transcript = "\n".join(f"[{e.speaker}] {e.text}" for e in expert_entries)

        prompt = (
            f"Extract structured notes from these new transcript entries:\n\n{transcript}"
        )
        response = await self._call_model(prompt)
        parsed = self._safe_parse_json(response, [])

        notes_list = parsed if isinstance(parsed, list) else parsed.get("notes", [])

        actions = []
        latest = []
        for note_data in notes_list:
            try:
                note = StructuredNote.model_validate(note_data)
            except Exception:
                continue
            latest.append(note)
            actions.append(AgentAction(
                agent_name=self.name,
                action_type="note",
                content=note.content,
                metadata={
                    "category": note.category,
                    "confidence": note.confidence,
                    "source_quote": note.source_quote,
                },
            ))

        self._latest_notes = latest
        return actions

    def get_latest_notes(self) -> list[StructuredNote]:
        return self._latest_notes
```

- [ ] **Step 4: Replace `agents/orchestrator.py`**

Only change vs current: pass `temperature=self.temperature` (None → provider default for natural phrasing) and coerce a `None` response to `""`.

```python
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
```

- [ ] **Step 5: Verify imports + that QA can construct/return None shape**

Run: `cd expert_call_agent && python3 -c "from agents.qa_agent import QAAgent; from agents.followup_agent import FollowUpAgent; from agents.note_taker import NoteTakerAgent; from agents.orchestrator import OrchestratorAgent; print('ok')"`
Expected: `ok`

- [ ] **Step 6: Commit**

```bash
git add expert_call_agent/agents/qa_agent.py expert_call_agent/agents/followup_agent.py expert_call_agent/agents/note_taker.py expert_call_agent/agents/orchestrator.py
git commit -m "agents: QA may stay silent; note-taker filters to expert turns; safe parse + low temperature; drop unused priority"
```

---

## Task 6: LiveCallQueue — tick/TTL/cooldown/near-dup dedup (issues #4, #5, #8)

**Files:**
- Modify: `expert_call_agent/live_orchestration.py` (full replace)

The queue gains a turn counter (`tick()` once per expert turn), a per-flag TTL, a post-speak cooldown that only urgent tiers can bypass, and near-duplicate (token-set Jaccard) dedup against both already-asked and still-pending flags. Selection remains fully deterministic given a fixed input set.

- [ ] **Step 1: Replace `live_orchestration.py`**

```python
from models import AgentAction
import config


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace, for dedup comparisons."""
    return " ".join((text or "").lower().split())


def _tokens(text: str) -> set[str]:
    return set(_normalize(text).split())


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


class LiveCallQueue:
    """Deterministic priority queue for live-call flags.

    Flags are ordered by a fixed tier hierarchy (``config.FLAG_TIER_ORDER``);
    ties break FIFO (insertion order). Selection is fully deterministic given a
    fixed set of input flags.

    Beyond tiering, the queue enforces four live-call policies:

    * **Dedup** — exact and near-duplicate (token-set Jaccard >=
      ``config.LIVE_DEDUP_JACCARD``) flags are never queued twice, whether
      against already-asked or still-pending flags.
    * **Per-type caps** — each flag type may be surfaced at most
      ``config.LIVE_FLAG_CAPS`` times per call (absent = unlimited).
    * **TTL** — a pending flag not surfaced within
      ``config.LIVE_FLAG_TTL_TURNS`` expert turns is dropped as stale.
    * **Cooldown** — after the AI speaks it stays quiet for
      ``config.LIVE_MIN_TURNS_BETWEEN_QUESTIONS`` turns, except for urgent tiers
      in ``config.LIVE_COOLDOWN_EXEMPT_TIERS``.

    Call ``tick()`` once per expert turn, before ``enqueue``/``select_next``.
    """

    def __init__(self):
        self._pending: list[tuple[int, int, AgentAction]] = []  # (seq, turn, action)
        self._asked: set[str] = set()
        self._asked_tokens: list[set[str]] = []
        self._asked_counts: dict[str, int] = {}
        self._seq = 0
        self._turn = 0
        self._last_spoke_turn = -(10 ** 9)

    def tick(self) -> None:
        """Advance the turn counter; call once per processed expert entry."""
        self._turn += 1

    def _tier_rank(self, action: AgentAction) -> int:
        try:
            return config.FLAG_TIER_ORDER.index(action.flag_type)
        except ValueError:
            return len(config.FLAG_TIER_ORDER)  # unknown types sort last

    def _at_cap(self, flag_type: str) -> bool:
        cap = config.LIVE_FLAG_CAPS.get(flag_type)
        return cap is not None and self._asked_counts.get(flag_type, 0) >= cap

    def _is_duplicate(self, key: str, key_tokens: set[str]) -> bool:
        if key in self._asked:
            return True
        for toks in self._asked_tokens:
            if _jaccard(key_tokens, toks) >= config.LIVE_DEDUP_JACCARD:
                return True
        for _, _, a in self._pending:
            other = _normalize(a.content)
            if other == key or _jaccard(key_tokens, _tokens(a.content)) >= config.LIVE_DEDUP_JACCARD:
                return True
        return False

    def enqueue(self, actions: list[AgentAction]) -> None:
        for action in actions:
            key = _normalize(action.content)
            if not key:
                continue
            if self._at_cap(action.flag_type):
                continue
            key_tokens = _tokens(action.content)
            if self._is_duplicate(key, key_tokens):
                continue
            self._pending.append((self._seq, self._turn, action))
            self._seq += 1

    def select_next(self) -> AgentAction | None:
        # Drop flags that are capped or stale (TTL exceeded).
        self._pending = [
            (seq, turn, a)
            for seq, turn, a in self._pending
            if not self._at_cap(a.flag_type)
            and (self._turn - turn) <= config.LIVE_FLAG_TTL_TURNS
        ]
        if not self._pending:
            return None

        in_cooldown = (
            self._turn - self._last_spoke_turn
        ) < config.LIVE_MIN_TURNS_BETWEEN_QUESTIONS

        candidates = [
            (seq, turn, a)
            for seq, turn, a in self._pending
            if not in_cooldown or a.flag_type in config.LIVE_COOLDOWN_EXEMPT_TIERS
        ]
        if not candidates:
            return None

        seq, turn, action = min(
            candidates, key=lambda c: (self._tier_rank(c[2]), c[0])
        )
        self._pending.remove((seq, turn, action))

        key = _normalize(action.content)
        self._asked.add(key)
        self._asked_tokens.append(_tokens(action.content))
        self._asked_counts[action.flag_type] = (
            self._asked_counts.get(action.flag_type, 0) + 1
        )
        self._last_spoke_turn = self._turn
        return action
```

- [ ] **Step 2: Verify import**

Run: `cd expert_call_agent && python3 -c "from live_orchestration import LiveCallQueue; LiveCallQueue(); print('ok')"`
Expected: `ok`
(Full behavioral check happens in Task 9.)

- [ ] **Step 3: Commit**

```bash
git add expert_call_agent/live_orchestration.py
git commit -m "live_orchestration: add turn TTL, post-speak cooldown, and near-duplicate dedup"
```

---

## Task 7: models.py + session.py — dead-code sweep + single-connection guard (issues #3, dead code)

**Files:**
- Modify: `expert_call_agent/models.py` (two targeted edits)
- Modify: `expert_call_agent/session.py` (full replace)

- [ ] **Step 1: Remove the unused `priority` field from `AgentAction` in `models.py`**

Replace:

```python
class AgentAction(BaseModel):
    agent_name: str
    action_type: str
    flag_type: str = "should_ask"  # contradiction | must_ask | probe | should_ask | nice_to_have
    content: str
    priority: float = 0.5  # retained for future same-tier tiebreak
    metadata: dict = {}
```

with:

```python
class AgentAction(BaseModel):
    agent_name: str
    action_type: str
    flag_type: str = "should_ask"  # contradiction | must_ask | probe | should_ask | nice_to_have
    content: str
    metadata: dict = {}
```

- [ ] **Step 2: Remove the unused `pending_actions` field from `CallSession` in `models.py`**

Replace:

```python
class CallSession(BaseModel):
    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "pre_call"
    project_context: ProjectContext | None = None
    call_guide: CallGuide | None = None
    transcript: list[TranscriptEntry] = []
    notes: list[StructuredNote] = []
    coverage: list[CoverageStatus] = []
    key_takeaways: list[str] = []
    pending_actions: list[AgentAction] = []
    final_summary: str | None = None
```

with:

```python
class CallSession(BaseModel):
    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "pre_call"
    project_context: ProjectContext | None = None
    call_guide: CallGuide | None = None
    transcript: list[TranscriptEntry] = []
    notes: list[StructuredNote] = []
    coverage: list[CoverageStatus] = []
    key_takeaways: list[str] = []
    final_summary: str | None = None
```

- [ ] **Step 3: Replace `session.py`**

Removes the dead `add_action`, `pop_pending_actions`, and `update_takeaways` (and the now-unused `AgentAction` import); adds `acquire_call`/`release_call` for the single-connection guard.

```python
import asyncio
from models import CallSession, TranscriptEntry, StructuredNote, CoverageStatus


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, CallSession] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._active_calls: set[str] = set()

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]

    async def create_session(self) -> CallSession:
        session = CallSession()
        self._sessions[session.session_id] = session
        return session

    async def get_session(self, session_id: str) -> CallSession:
        if session_id not in self._sessions:
            raise KeyError(f"Session {session_id} not found")
        return self._sessions[session_id]

    async def update_session(self, session_id: str, **updates) -> CallSession:
        async with self._get_lock(session_id):
            session = self._sessions[session_id]
            for key, value in updates.items():
                setattr(session, key, value)
            return session

    async def add_transcript_entry(self, session_id: str, entry: TranscriptEntry):
        async with self._get_lock(session_id):
            self._sessions[session_id].transcript.append(entry)

    async def add_note(self, session_id: str, note: StructuredNote):
        async with self._get_lock(session_id):
            self._sessions[session_id].notes.append(note)

    async def update_coverage(self, session_id: str, coverage: list[CoverageStatus]):
        async with self._get_lock(session_id):
            self._sessions[session_id].coverage = coverage

    async def acquire_call(self, session_id: str) -> bool:
        """Claim the single live-call slot for a session. False if already active."""
        async with self._get_lock(session_id):
            if session_id in self._active_calls:
                return False
            self._active_calls.add(session_id)
            return True

    async def release_call(self, session_id: str) -> None:
        async with self._get_lock(session_id):
            self._active_calls.discard(session_id)
```

- [ ] **Step 4: Verify imports + dead names are gone**

Run: `cd expert_call_agent && python3 -c "import models, session; m=models.AgentAction(agent_name='a',action_type='t',content='c'); assert not hasattr(m,'priority'); s=session.SessionManager(); assert hasattr(s,'acquire_call') and not hasattr(s,'add_action'); print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add expert_call_agent/models.py expert_call_agent/session.py
git commit -m "models/session: drop dead AgentAction.priority + pending_actions/add_action/update_takeaways; add single-connection guard"
```

---

## Task 8: routes_call.py — timeouts, connection guard, reconnect seeding, queue tick, single fetch (issues #1, #3)

**Files:**
- Modify: `expert_call_agent/api/routes_call.py` (full replace)

Wraps every live agent call in `asyncio.wait_for` (timeouts surface as captured exceptions with informative messages); claims/releases the single-connection slot; seeds the note-taker so a reconnect doesn't re-extract existing notes; calls `queue.tick()` once per turn; fetches the session once per `process_entry`; drops the unused `import time`. The `DEMO_EXPERT_RESPONSES` list is unchanged — **keep its contents exactly as currently committed.**

- [ ] **Step 1: Replace `api/routes_call.py`** (keep the existing `DEMO_EXPERT_RESPONSES` contents verbatim where indicated)

```python
import asyncio
import base64
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, HTTPException

from agents.qa_agent import QAAgent
from agents.followup_agent import FollowUpAgent
from agents.note_taker import NoteTakerAgent
from agents.orchestrator import OrchestratorAgent
from live_orchestration import LiveCallQueue
from models import TranscriptEntry, AgentAction
import config

router = APIRouter()

DEMO_EXPERT_RESPONSES = [
    # >>> KEEP the existing 10 strings from the current file exactly as-is <<<
]


def _err(exc: Exception) -> str:
    msg = str(exc)
    return f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__


async def _run_agent(coro):
    """Run a live-call agent with a hard timeout so a hung LLM call can't stall the loop."""
    return await asyncio.wait_for(coro, timeout=config.AGENT_TIMEOUT_SECONDS)


@router.post("/start/{session_id}")
async def start_call(request: Request, session_id: str):
    sessions = request.app.state.sessions
    session = await sessions.get_session(session_id)
    if not session.call_guide:
        raise HTTPException(400, "Generate a call guide first")
    await sessions.update_session(session_id, status="live")
    return {"status": "live", "session_id": session_id}


@router.post("/end/{session_id}")
async def end_call(request: Request, session_id: str):
    sessions = request.app.state.sessions
    await sessions.update_session(session_id, status="post_call")
    return {"status": "post_call", "session_id": session_id}


@router.websocket("/ws/{session_id}")
async def call_websocket(ws: WebSocket, session_id: str):
    await ws.accept()

    app = ws.app
    sessions = app.state.sessions
    gemini = app.state.gemini
    claude = app.state.claude
    stt = app.state.stt

    try:
        session = await sessions.get_session(session_id)
    except KeyError:
        await ws.send_json({"type": "error", "message": "Session not found"})
        await ws.close()
        return

    # One live connection per session: a reconnect / second tab must not run a
    # parallel loop against shared session state.
    if not await sessions.acquire_call(session_id):
        await ws.send_json({"type": "error", "message": "Call already active for this session"})
        await ws.close()
        return

    try:
        qa = QAAgent(gemini, claude)
        followup = FollowUpAgent(gemini, claude)
        note_taker = NoteTakerAgent(gemini, claude)
        orchestrator = OrchestratorAgent(gemini, claude)
        queue = LiveCallQueue()

        # On reconnect, don't re-extract notes for transcript entries already recorded.
        note_taker.seed_processed(len(session.transcript))

        async def process_entry(entry: TranscriptEntry):
            await sessions.add_transcript_entry(session_id, entry)
            session = await sessions.get_session(session_id)
            await ws.send_json({"type": "transcript", "entry": entry.model_dump()})

            qa_result, fu_result, nt_result = await asyncio.gather(
                _run_agent(qa.run(session)),
                _run_agent(followup.run(session)),
                _run_agent(note_taker.run(session)),
                return_exceptions=True,
            )

            # Note-taker (passive) — record notes, never queued.
            if isinstance(nt_result, Exception):
                await ws.send_json({"type": "error", "message": _err(nt_result)})
            else:
                for note in note_taker.get_latest_notes():
                    await sessions.add_note(session_id, note)
                    await ws.send_json({"type": "note", "note": note.model_dump()})

            # Coverage status from the Follow-up agent.
            if isinstance(fu_result, Exception):
                await ws.send_json({"type": "error", "message": _err(fu_result)})
            else:
                coverage = followup.get_latest_coverage()
                if coverage:
                    await sessions.update_coverage(session_id, coverage)
                    await ws.send_json({
                        "type": "coverage_update",
                        "coverage": [c.model_dump() for c in coverage],
                    })

            # Collect flags from QA + Follow-up.
            flags: list[AgentAction] = []
            if isinstance(qa_result, AgentAction):
                flags.append(qa_result)
            elif isinstance(qa_result, Exception):
                await ws.send_json({"type": "error", "message": _err(qa_result)})
            if isinstance(fu_result, list):
                flags.extend(fu_result)

            # Deterministic selection.
            queue.tick()
            queue.enqueue(flags)
            selected = queue.select_next()
            if selected is None:
                return  # nothing worth raising -> stay silent

            # Compose the spoken turn.
            try:
                utterance = await _run_agent(orchestrator.run(session, flag=selected))
            except Exception as e:
                await ws.send_json({"type": "error", "message": _err(e)})
                return
            if not utterance or not utterance.strip():
                return
            utterance = utterance.strip()

            await ws.send_json({
                "type": "ai_turn",
                "question": utterance,
                "agent": selected.agent_name,
                "flag_type": selected.flag_type,
                "rationale": selected.metadata.get("rationale") or selected.metadata.get("reason", ""),
            })

            ai_entry = TranscriptEntry(speaker="interviewer", text=utterance)
            await sessions.add_transcript_entry(session_id, ai_entry)
            await ws.send_json({"type": "transcript", "entry": ai_entry.model_dump()})

            # Auto-speak via TTS.
            try:
                audio_bytes = await app.state.tts.synthesize(utterance)
                await ws.send_json({
                    "type": "tts_audio",
                    "data": base64.b64encode(audio_bytes).decode(),
                })
            except Exception:
                pass

        while True:
            raw = await ws.receive()
            if raw.get("type") == "websocket.disconnect":
                break

            if "bytes" in raw:
                audio = raw["bytes"]
                if len(audio) < 100:
                    continue
                try:
                    text = await stt.transcribe(audio, mime_type="audio/webm")
                    text = text.strip()
                    if text:
                        entry = TranscriptEntry(speaker="expert", text=text)
                        await process_entry(entry)
                except Exception as e:
                    await ws.send_json({"type": "error", "message": f"STT: {_err(e)}"})

            elif "text" in raw:
                data = json.loads(raw["text"])

                if data.get("type") == "text_input":
                    entry = TranscriptEntry(
                        speaker=data.get("speaker", "expert"),
                        text=data["text"],
                    )
                    await process_entry(entry)

                elif data.get("type") == "run_demo":
                    for i, expert_text in enumerate(DEMO_EXPERT_RESPONSES):
                        await ws.send_json({
                            "type": "demo_progress",
                            "current": i + 1,
                            "total": len(DEMO_EXPERT_RESPONSES),
                        })
                        entry = TranscriptEntry(speaker="expert", text=expert_text)
                        await process_entry(entry)
                        await asyncio.sleep(1)

                    await ws.send_json({"type": "demo_complete"})

    except WebSocketDisconnect:
        pass
    finally:
        await sessions.release_call(session_id)
```

- [ ] **Step 2: Restore `DEMO_EXPERT_RESPONSES` contents**

The placeholder comment above must be replaced with the **exact 10 strings** from the previously committed `routes_call.py` (recover from git if needed: `git show HEAD:expert_call_agent/api/routes_call.py`). Do not paraphrase them.

- [ ] **Step 3: Verify it imports and the app builds**

Run: `cd expert_call_agent && python3 -c "from api.routes_call import router, _err, _run_agent; print(len(__import__('api.routes_call', fromlist=['DEMO_EXPERT_RESPONSES']).DEMO_EXPERT_RESPONSES), 'demo responses')"`
Expected: `10 demo responses`

Run: `cd expert_call_agent && python3 -c "from api.app import create_app; create_app(); print('app builds')"`
Expected: `app builds`

- [ ] **Step 4: Commit**

```bash
git add expert_call_agent/api/routes_call.py
git commit -m "routes_call: per-call agent timeouts, single-connection guard, reconnect note seeding, queue tick"
```

---

## Task 9: Verification (no committed tests)

**Files:** none committed — ad-hoc scripts only.

- [ ] **Step 1: Byte-compile the whole package**

Run:
```bash
cd expert_call_agent && python3 -m py_compile $(git ls-files '*.py')
```
Expected: no output, exit code 0.

- [ ] **Step 2: Behavioral check of `LiveCallQueue` (pure logic, no network)**

Run:
```bash
cd expert_call_agent && python3 - <<'PY'
import config
from models import AgentAction
from live_orchestration import LiveCallQueue

def mk(ft, content):
    return AgentAction(agent_name="x", action_type="t", flag_type=ft, content=content)

# 1) Tier order: contradiction wins over should_ask in the same turn.
q = LiveCallQueue()
q.tick()
q.enqueue([mk("should_ask", "What is revenue?"), mk("contradiction", "Margins 60 vs 45, clarify?")])
assert q.select_next().flag_type == "contradiction"

# 2) Cooldown: a non-urgent flag right after speaking is suppressed (gap < 2)...
q.tick()
q.enqueue([mk("should_ask", "How many GPUs are deployed?")])
assert q.select_next() is None
# ...then surfaced once the cooldown passes.
q.tick()
assert q.select_next() is not None

# 3) Near-duplicate suppression (token-set Jaccard >= 0.8).
q2 = LiveCallQueue()
q2.tick()
q2.enqueue([mk("should_ask", "What is the total revenue")])
assert q2.select_next() is not None
q2.tick(); q2.tick()
q2.enqueue([mk("should_ask", "What is the total revenue figure")])
assert q2.select_next() is None  # near-dup of an already-asked question

# 4) TTL: a low-priority flag starved past its TTL is pruned.
q3 = LiveCallQueue()
q3.tick()
q3.enqueue([mk("nice_to_have", "Low priority aside")])
for i in range(config.LIVE_FLAG_TTL_TURNS + 1):
    q3.tick()
    q3.enqueue([mk("contradiction", f"Conflict {i}?")])  # always wins, starves the aside
    q3.select_next()
assert q3.select_next() is None  # aside is now stale and pruned

print("QUEUE OK")
PY
```
Expected: `QUEUE OK`

- [ ] **Step 3: Frontend contract unchanged — JS syntax check**

Run:
```bash
node --check expert_call_agent/static/index.html 2>/dev/null || echo "NOTE: index.html is not pure JS; instead grep the handled message types"
grep -oE "ai_turn|tts_audio|coverage_update|demo_progress|demo_complete|transcript|note" expert_call_agent/static/index.html | sort -u
```
Expected: the grep lists the same message-type strings the backend still emits (no new/removed types). The backend contract did not change in this plan, so this is a sanity check only.

- [ ] **Step 4 (OPTIONAL — uses Vertex/ADC + real API calls): end-to-end smoke**

Only run if ADC is configured and spending a few API calls is acceptable. This drives the real WebSocket loop via FastAPI's in-process `TestClient` (no server/tunnel needed).

```bash
cd expert_call_agent && python3 - <<'PY'
from fastapi.testclient import TestClient
from api.app import create_app

app = create_app()
with TestClient(app) as client:
    sid = client.post("/api/precall/session").json()["session_id"] \
        if False else None  # adjust to the actual precall session-create route if different
PY
```
**Note:** the precall session-create route name must match `api/routes_precall.py`. Inspect that file first and adapt the smoke (upload brief → generate guide → open `/api/call/ws/{sid}` → send `{"type":"run_demo"}` → assert you receive `ai_turn` + `tts_audio` messages and `demo_complete`, with **fewer** `ai_turn`s than expert turns now that the cooldown is active). If running this is not desired, rely on Steps 1–3 plus a manual UI walkthrough via the tunnel.

- [ ] **Step 5: Final commit (only if Step 3/4 surfaced any needed tweak)**

```bash
git add -A && git commit -m "fix: address live-call hardening verification findings"
```

---

## Self-review checklist (performed by plan author)

- **Spec coverage:** every issue #1–#10 and each Low item maps to a task in the traceability table; deferred items are listed under "Out of scope" with rationale. ✓
- **Type consistency:** `AgentAction` no longer has `priority` (T7) and no task constructs it with `priority=` (T5). `note_taker.seed_processed()` (T5) is the exact name called in T8. `queue.tick()`/`enqueue()`/`select_next()` (T6) match the call sites in T8. `get_auth()` (T2) matches its callers in T2. `_run_agent`/`_err` (T8) defined before use. ✓
- **No placeholders:** all code steps contain complete code. The one deliberate placeholder — `DEMO_EXPERT_RESPONSES` contents — has an explicit "restore verbatim from git" step (T8 Step 2) because reproducing 10 long canned strings risks transcription drift. ✓
- **Contract safety:** no server→client or client→server message type added/removed/renamed; the frontend needs no changes. ✓

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-31-live-call-orchestration-hardening.md`. You mentioned handing this to a build agent — the plan is self-contained per task (full code inline), so dispatching one task at a time with a review between tasks (subagent-driven-development) will work cleanly. Recommended order is T1 → T9 as written (later tasks depend on earlier ones: agents need config/base, routes need session/queue/agents).
