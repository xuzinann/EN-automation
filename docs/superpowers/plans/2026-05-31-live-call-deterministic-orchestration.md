# Live-Call Deterministic Orchestration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the LLM orchestrator-selector with a deterministic priority queue, recast the Orchestrator as an autonomous responder (composes + speaks via TTS), and merge live + post-call summarization into one post-call agent.

**Architecture:** Per expert turn, QA and Follow-up agents emit *flags* (typed by role). A deterministic priority queue (`live_orchestration.py`) orders flags by a fixed tier hierarchy (FIFO tiebreak, dedup of asked questions) and selects one. The Orchestrator composes that flag into a natural utterance, which is auto-spoken via Google Cloud TTS. The Note-taker runs passively. Post-call, a single Claude summarizer emits structured takeaways + a full markdown report.

**Tech Stack:** Python 3.12, FastAPI + WebSockets, Pydantic v2, Vertex AI (Gemini via `google-genai`, Claude via `rawPredict`), Google Cloud TTS, vanilla-JS frontend.

**Spec:** `docs/superpowers/specs/2026-05-31-live-call-deterministic-orchestration-design.md`

**Testing note (hackathon):** Per the user, detailed automated tests are intentionally skipped. Verification is by running the app / demo and observing behavior, plus one tiny inline sanity check for the queue. No `pytest` dependency is added.

**Working directory:** All paths are relative to `expert_call_agent/`. Module imports are top-level (e.g. `import config`, `from models import ...`), so commands run from inside `expert_call_agent/`.

---

## Setup (once, before Task 1)

- [ ] **Step 1: Install runtime deps**

Run (from repo root):
```bash
cd expert_call_agent && pip install -r requirements.txt
```
Expected: installs fastapi, uvicorn, httpx, pydantic, google-genai, google-auth, aiofiles (or "already satisfied").

- [ ] **Step 2: Confirm the app imports**

Run (from `expert_call_agent/`):
```bash
python3 -c "import models, config; print('imports ok')"
```
Expected: `imports ok`

---

## Task 1: Data model + config

Add the `flag_type` field to `AgentAction`, define the fixed tier order, and drop the deleted live summarizer from the model registry.

**Files:**
- Modify: `expert_call_agent/models.py` (the `AgentAction` class)
- Modify: `expert_call_agent/config.py`

- [ ] **Step 1: Add `flag_type` to `AgentAction`**

In `models.py`, replace the `AgentAction` class:
```python
class AgentAction(BaseModel):
    agent_name: str
    action_type: str
    flag_type: str = "should_ask"  # contradiction | must_ask | probe | should_ask | nice_to_have
    content: str
    priority: float = 0.5  # retained for future same-tier tiebreak
    metadata: dict = {}
```

- [ ] **Step 2: Update `config.py` — drop summarizer, add tier order**

In `config.py`, replace the `AGENT_MODELS` dict and append the tier constant:
```python
AGENT_MODELS = {
    "context_ingestion": ("claude", CLAUDE_MODEL),
    "call_guide_drafter": ("claude", CLAUDE_MODEL),
    "orchestrator": ("gemini", GEMINI_FLASH_MODEL),
    "qa_agent": ("claude", CLAUDE_MODEL),
    "followup_agent": ("gemini", GEMINI_PRO_MODEL),
    "note_taker": ("gemini", GEMINI_FLASH_MODEL),
    "post_call_summarizer": ("claude", CLAUDE_MODEL),
}

ORCHESTRATOR_TICK_SECONDS = 5
MAX_TRANSCRIPT_CONTEXT = 20

# Live-call flag priority: lower index = higher priority. FIFO breaks ties.
FLAG_TIER_ORDER = ["contradiction", "must_ask", "probe", "should_ask", "nice_to_have"]
```

- [ ] **Step 3: Verify imports still work**

Run (from `expert_call_agent/`):
```bash
python3 -c "import config, models; print(config.FLAG_TIER_ORDER); print(models.AgentAction(agent_name='x', action_type='y', content='z').flag_type)"
```
Expected: prints the tier list, then `should_ask`.

- [ ] **Step 4: Commit**

```bash
git add expert_call_agent/models.py expert_call_agent/config.py
git commit -m "feat: add flag_type + FLAG_TIER_ORDER, drop live summarizer from registry"
```

---

## Task 2: Deterministic priority queue

Create the pure-code queue that orders flags by tier, breaks ties FIFO, dedups already-asked questions, and pops one flag per turn. This is the heart of the change — verify it with a quick inline sanity check.

**Files:**
- Create: `expert_call_agent/live_orchestration.py`

- [ ] **Step 1: Write the queue**

Create `live_orchestration.py`:
```python
from models import AgentAction
import config


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace, for dedup comparisons."""
    return " ".join((text or "").lower().split())


class LiveCallQueue:
    """Deterministic priority queue for live-call flags.

    Flags are ordered by a fixed tier hierarchy (``config.FLAG_TIER_ORDER``);
    ties break FIFO (insertion order). Selected flags are remembered so the
    same question is never surfaced twice. Unselected flags persist across
    turns until they are selected or re-supersede.
    """

    def __init__(self):
        self._pending: list[tuple[int, AgentAction]] = []
        self._asked: set[str] = set()
        self._seq = 0

    def _tier_rank(self, action: AgentAction) -> int:
        try:
            return config.FLAG_TIER_ORDER.index(action.flag_type)
        except ValueError:
            return len(config.FLAG_TIER_ORDER)  # unknown types sort last

    def enqueue(self, actions: list[AgentAction]) -> None:
        for action in actions:
            key = _normalize(action.content)
            if not key or key in self._asked:
                continue
            if any(_normalize(a.content) == key for _, a in self._pending):
                continue
            self._pending.append((self._seq, action))
            self._seq += 1

    def select_next(self) -> AgentAction | None:
        if not self._pending:
            return None
        idx = min(
            range(len(self._pending)),
            key=lambda i: (self._tier_rank(self._pending[i][1]), self._pending[i][0]),
        )
        _, action = self._pending.pop(idx)
        self._asked.add(_normalize(action.content))
        return action
```

- [ ] **Step 2: Sanity-check the queue (inline, no test file)**

Run (from `expert_call_agent/`):
```bash
python3 -c "
from live_orchestration import LiveCallQueue
from models import AgentAction
def a(ft, c): return AgentAction(agent_name='t', action_type='x', flag_type=ft, content=c)
q = LiveCallQueue()
q.enqueue([a('should_ask','Q1'), a('contradiction','C1'), a('must_ask','M1')])
order = [q.select_next().content for _ in range(3)]
assert order == ['C1','M1','Q1'], order        # tier order
assert q.select_next() is None                  # empty -> None
q.enqueue([a('must_ask','M1')]); assert q.select_next() is None   # asked dedup
q.enqueue([a('probe','P1'), a('probe','P2')])
assert [q.select_next().content for _ in range(2)] == ['P1','P2']  # FIFO within tier
print('queue ok')
"
```
Expected: `queue ok` (no AssertionError).

- [ ] **Step 3: Commit**

```bash
git add expert_call_agent/live_orchestration.py
git commit -m "feat: deterministic live-call priority queue"
```

---

## Task 3: QA agent → flagger

Make the QA agent return the matched question's priority so its flag carries a `flag_type`.

**Files:**
- Modify: `expert_call_agent/agents/qa_agent.py`

- [ ] **Step 1: Update the QA agent**

Replace the contents of `agents/qa_agent.py`:
```python
from agents.base_agent import BaseAgent
from models import CallSession, AgentAction
import config

_VALID_FLAGS = {"must_ask", "should_ask", "nice_to_have"}


class QAAgent(BaseAgent):
    name = "qa_agent"
    system_prompt = (
        "You are the lead interviewer in a PE expert call. "
        "Given the interview guide and real-time transcript, suggest the best next "
        "question to ask. Adapt based on what the expert has said — don't repeat "
        "questions already answered, and probe deeper when answers are vague.\n\n"
        "Return ONLY valid JSON:\n"
        '{"question": "the question to ask", "rationale": "why this question now", '
        '"guide_section": "which section this addresses", '
        '"priority": "must_ask|should_ask|nice_to_have"}\n'
        "Set priority from the guide question's own priority (default should_ask).\n"
        "No markdown fences or commentary."
    )

    async def run(self, session: CallSession, **kwargs) -> AgentAction:
        guide_json = session.call_guide.model_dump_json() if session.call_guide else "{}"
        recent = session.transcript[-config.MAX_TRANSCRIPT_CONTEXT:]
        transcript = "\n".join(f"[{e.speaker}] {e.text}" for e in recent)

        prompt = (
            f"Interview guide:\n{guide_json}\n\n"
            f"Transcript so far:\n{transcript}\n\n"
            "Suggest the single best next question to ask the expert."
        )
        response = await self._call_model(prompt)
        parsed = self._parse_json(response)

        flag_type = parsed.get("priority", "should_ask")
        if flag_type not in _VALID_FLAGS:
            flag_type = "should_ask"

        return AgentAction(
            agent_name=self.name,
            action_type="suggest_question",
            flag_type=flag_type,
            content=parsed.get("question", response),
            priority=0.8,
            metadata={
                "rationale": parsed.get("rationale", ""),
                "section": parsed.get("guide_section", ""),
            },
        )
```

- [ ] **Step 2: Verify it imports**

Run (from `expert_call_agent/`):
```bash
python3 -c "from agents.qa_agent import QAAgent; print('qa ok')"
```
Expected: `qa ok`

- [ ] **Step 3: Commit**

```bash
git add expert_call_agent/agents/qa_agent.py
git commit -m "feat: QA agent emits typed flag (must_ask/should_ask/nice_to_have)"
```

---

## Task 4: Follow-up agent → flagger

Tag the Follow-up agent's emitted actions with `flag_type` (`contradiction` for conflicts, `probe` for follow-ups). Coverage output is unchanged.

**Files:**
- Modify: `expert_call_agent/agents/followup_agent.py` (the `run` method's action construction)

- [ ] **Step 1: Add `flag_type` to the follow-up flags**

In `agents/followup_agent.py`, replace the two action-building loops inside `run` (the `for fu in ...` and `for contradiction in ...` blocks) with:
```python
        actions = []
        for fu in parsed.get("followups", []):
            actions.append(AgentAction(
                agent_name=self.name,
                action_type="followup",
                flag_type="probe",
                content=fu.get("question", ""),
                priority=fu.get("priority", 0.6),
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
                priority=0.9,
                metadata={"reason": "contradiction with known data"},
            ))
```

- [ ] **Step 2: Verify it imports**

Run (from `expert_call_agent/`):
```bash
python3 -c "from agents.followup_agent import FollowUpAgent; print('followup ok')"
```
Expected: `followup ok`

- [ ] **Step 3: Commit**

```bash
git add expert_call_agent/agents/followup_agent.py
git commit -m "feat: Follow-up agent tags flags (contradiction/probe)"
```

---

## Task 5: Orchestrator → responder

Rewrite the Orchestrator from an LLM selector into a responder: given one chosen flag + recent transcript, compose a single natural spoken turn (plain text, not JSON).

**Files:**
- Modify: `expert_call_agent/agents/orchestrator.py` (full rewrite)

- [ ] **Step 1: Rewrite the orchestrator**

Replace the entire contents of `agents/orchestrator.py`:
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
        # Orchestrator is always the Gemini Flash provider (see config.AGENT_MODELS);
        # call it directly for a plain-text (non-JSON) response.
        return await self.gemini.generate(
            model=self.model_name,
            prompt=prompt,
            system_instruction=self.system_prompt,
        )
```

- [ ] **Step 2: Verify it imports**

Run (from `expert_call_agent/`):
```bash
python3 -c "from agents.orchestrator import OrchestratorAgent; print('orchestrator ok')"
```
Expected: `orchestrator ok`

- [ ] **Step 3: Commit**

```bash
git add expert_call_agent/agents/orchestrator.py
git commit -m "feat: rewrite Orchestrator as autonomous responder (composes one spoken turn)"
```

---

## Task 6: Merged post-call summarizer

Fold the live Key-Takeaway Summarizer into the post-call summarizer: one call returns structured takeaways + the full markdown report. Delete the old live summarizer and update the post-call route.

**Files:**
- Modify: `expert_call_agent/agents/post_call_summarizer.py` (full rewrite)
- Delete: `expert_call_agent/agents/summarizer.py`
- Modify: `expert_call_agent/api/routes_postcall.py` (the `summarize` handler)

- [ ] **Step 1: Rewrite the post-call summarizer**

Replace the entire contents of `agents/post_call_summarizer.py`:
```python
import json
import re
from agents.base_agent import BaseAgent
from models import CallSession


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
```

- [ ] **Step 2: Delete the live summarizer**

Run:
```bash
git rm expert_call_agent/agents/summarizer.py
```
Expected: `rm 'expert_call_agent/agents/summarizer.py'`

- [ ] **Step 3: Update the post-call route**

In `api/routes_postcall.py`, replace the `summarize` handler body's tail (from `agent = PostCallSummarizerAgent(...)` to the `return`) with:
```python
    agent = PostCallSummarizerAgent(gemini, claude)
    result = await agent.run(session)
    await sessions.update_session(
        session_id,
        final_summary=result["markdown"],
        key_takeaways=result["takeaways"],
        status="post_call",
    )

    return {"summary": result["markdown"], "takeaways": result["takeaways"]}
```

- [ ] **Step 4: Verify imports**

Run (from `expert_call_agent/`):
```bash
python3 -c "from agents.post_call_summarizer import PostCallSummarizerAgent; print('summarizer ok')"
python3 -c "import api.routes_postcall; print('route ok')"
```
Expected: `summarizer ok` then `route ok`.

- [ ] **Step 5: Commit**

```bash
git add expert_call_agent/agents/post_call_summarizer.py expert_call_agent/api/routes_postcall.py
git commit -m "feat: merge takeaways into post-call summarizer; delete live summarizer"
```

---

## Task 7: Rewrite the live WebSocket loop

Wire the new pieces together: parallel flaggers + passive note-taker → deterministic queue → responder → auto-TTS. Emit the new `ai_turn` message, drop `suggested_question` and the `ask_question` handler, and remove the live summarizer.

**Files:**
- Modify: `expert_call_agent/api/routes_call.py`

- [ ] **Step 1: Update imports**

In `api/routes_call.py`, replace the agent-import block (lines importing the agents + models) with:
```python
from agents.qa_agent import QAAgent
from agents.followup_agent import FollowUpAgent
from agents.note_taker import NoteTakerAgent
from agents.orchestrator import OrchestratorAgent
from live_orchestration import LiveCallQueue
from models import TranscriptEntry, AgentAction
```
(Note: the `KeyTakeawaySummarizerAgent` import is removed.)

- [ ] **Step 2: Replace agent instantiation + `process_entry`**

In the `call_websocket` function, replace everything from `qa = QAAgent(...)` down to the end of the `process_entry` function definition (i.e. through the old `suggested_question` send block) with:
```python
    qa = QAAgent(gemini, claude)
    followup = FollowUpAgent(gemini, claude)
    note_taker = NoteTakerAgent(gemini, claude)
    orchestrator = OrchestratorAgent(gemini, claude)
    queue = LiveCallQueue()

    async def process_entry(entry: TranscriptEntry):
        await sessions.add_transcript_entry(session_id, entry)
        session = await sessions.get_session(session_id)
        await ws.send_json({"type": "transcript", "entry": entry.model_dump()})

        qa_result, fu_result, nt_result = await asyncio.gather(
            qa.run(session),
            followup.run(session),
            note_taker.run(session),
            return_exceptions=True,
        )

        # Note-taker (passive) — record notes, never queued.
        if isinstance(nt_result, Exception):
            await ws.send_json({"type": "error", "message": str(nt_result)})
        else:
            for note in note_taker.get_latest_notes():
                await sessions.add_note(session_id, note)
                await ws.send_json({"type": "note", "note": note.model_dump()})

        # Coverage status from the Follow-up agent.
        if not isinstance(fu_result, Exception):
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
            await ws.send_json({"type": "error", "message": str(qa_result)})
        if isinstance(fu_result, list):
            flags.extend(fu_result)
        elif isinstance(fu_result, Exception):
            await ws.send_json({"type": "error", "message": str(fu_result)})

        # Deterministic selection.
        queue.enqueue(flags)
        selected = queue.select_next()
        if selected is None:
            return  # empty queue -> stay silent, let the expert continue

        # Compose the spoken turn.
        session = await sessions.get_session(session_id)
        utterance = await orchestrator.run(session, flag=selected)
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
```

- [ ] **Step 3: Remove the `ask_question` handler**

In the message-receive loop, delete the entire `elif data.get("type") == "ask_question":` block (the one that builds an interviewer entry and synthesizes TTS on demand). Keep the `text_input`, audio-`bytes`, and `run_demo` branches.

- [ ] **Step 4: Verify it imports**

Run (from `expert_call_agent/`):
```bash
python3 -c "import api.routes_call; print('routes_call ok')"
```
Expected: `routes_call ok`

- [ ] **Step 5: Commit**

```bash
git add expert_call_agent/api/routes_call.py
git commit -m "feat: autonomous live loop — deterministic queue + responder + auto-TTS"
```

---

## Task 8: Frontend autonomous mode

Repurpose the suggestions panel into an AI-interviewer monitor, remove the manual "Ask This" path, drop the live takeaways subsection, and render takeaways in the post-call summary.

**Files:**
- Modify: `expert_call_agent/static/index.html`

- [ ] **Step 1: Rename the suggestions panel**

Replace:
```html
    <div class="call-panel" id="suggestions-panel">
      <h3>Agent Suggestions</h3>
      <div id="suggestions-list"><p style="color:#999; font-size:13px;">Suggestions will appear here during the call.</p></div>
    </div>
```
with:
```html
    <div class="call-panel" id="suggestions-panel">
      <h3>AI Interviewer</h3>
      <div id="suggestions-list"><p style="color:#999; font-size:13px;">The AI's questions will appear here as it conducts the call.</p></div>
    </div>
```

- [ ] **Step 2: Remove the live Key Takeaways subsection**

Delete this block from the Notes panel:
```html
      <div style="margin-top:10px;">
        <strong style="font-size:12px; text-transform:uppercase; color:#666;">Key Takeaways</strong>
        <div id="takeaways-list"></div>
      </div>
```

- [ ] **Step 3: Update the WS message handler**

In `handleWSMessage`, replace the `suggested_question` and `key_takeaways` cases. Change:
```javascript
    case 'suggested_question':
      addSuggestion(msg);
      break;
```
to:
```javascript
    case 'ai_turn':
      addAiTurn(msg);
      break;
```
and delete the `key_takeaways` case:
```javascript
    case 'key_takeaways':
      updateTakeaways(msg.takeaways);
      break;
```

- [ ] **Step 4: Replace `addSuggestion` with `addAiTurn`; drop `askQuestion` + `updateTakeaways`**

Replace the `addSuggestion` function:
```javascript
function addAiTurn(msg) {
  const list = document.getElementById('suggestions-list');
  if (list.querySelector('p')) list.innerHTML = '';

  const card = document.createElement('div');
  card.className = 'suggestion-card';
  card.innerHTML = `
    <div class="agent-tag">${msg.agent || 'agent'} — ${(msg.flag_type || '').replace('_',' ')}</div>
    <div class="question">${msg.question}</div>
    <div class="rationale">${msg.rationale || ''}</div>
  `;
  list.insertBefore(card, list.firstChild);
}
```
Then delete the now-unused `askQuestion` function:
```javascript
function askQuestion(q) {
  if (!ws) return;
  ws.send(JSON.stringify({ type: 'ask_question', question: q }));
}
```
and the now-unused `updateTakeaways` function:
```javascript
function updateTakeaways(takeaways) {
  const list = document.getElementById('takeaways-list');
  list.innerHTML = takeaways.map(t => `<div class="takeaway-item">${t}</div>`).join('');
}
```

- [ ] **Step 5: Render takeaways in the post-call summary**

In `generateSummary`, replace:
```javascript
    const data = await resp.json();
    document.getElementById('summary-panel').classList.remove('hidden');
    document.getElementById('summary-content').innerHTML = markdownToHtml(data.summary);
    setStatus('summary-status', 'Summary generated.');
```
with:
```javascript
    const data = await resp.json();
    document.getElementById('summary-panel').classList.remove('hidden');
    let html = '';
    if (data.takeaways && data.takeaways.length) {
      html += '<div style="margin-bottom:16px; padding:12px; background:#e8f5e9; border-radius:6px;">';
      html += '<strong style="font-size:12px; text-transform:uppercase; color:#2e7d32;">Key Takeaways</strong>';
      html += '<ul style="margin:8px 0 0 18px; font-size:13px;">';
      html += data.takeaways.map(t => `<li>${t}</li>`).join('');
      html += '</ul></div>';
    }
    html += markdownToHtml(data.summary);
    document.getElementById('summary-content').innerHTML = html;
    setStatus('summary-status', 'Summary generated.');
```

- [ ] **Step 6: Verify no stale references remain**

Run (from `expert_call_agent/`):
```bash
grep -nE "ask_question|addSuggestion|suggested_question|takeaways-list|updateTakeaways" static/index.html || echo "clean"
```
Expected: `clean` (no matches).

- [ ] **Step 7: Commit**

```bash
git add expert_call_agent/static/index.html
git commit -m "feat: frontend autonomous mode — AI interviewer monitor, post-call takeaways"
```

---

## Task 9: End-to-end verification

Run the app and the built-in CoreWeave demo to confirm the autonomous loop works against live models.

**Files:** none (manual verification)

- [ ] **Step 1: Confirm Google ADC is available**

Run:
```bash
python3 -c "import google.auth; c,p=google.auth.default(); print('ADC project:', p)"
```
Expected: prints a project (e.g. `gp-ct-sbox-sat-gcp0bg-darksoft`). If it errors, run `gcloud auth application-default login` first (type `! gcloud auth application-default login` in the session).

- [ ] **Step 2: Start the server**

Run (from `expert_call_agent/`):
```bash
python3 run.py
```
Expected: uvicorn serving on `http://0.0.0.0:8888`.

- [ ] **Step 3: Drive the demo in a browser**

Open `http://localhost:8888`. Click **Use Sample Brief** → wait for context → **Generate Call Guide** → **Run Demo Simulation**.

Confirm, as the demo plays:
- Expert turns appear in the Transcript.
- The **AI Interviewer** panel shows composed questions tagged with the driving agent + flag type.
- Interviewer turns appear in the Transcript and TTS audio auto-plays (if browser autoplay allows).
- Notes and Coverage populate; no live "Key Takeaways" panel exists.

- [ ] **Step 4: Generate the post-call summary**

After the demo completes, click **Generate Summary**. Confirm the panel shows a **Key Takeaways** list followed by the full markdown report.

- [ ] **Step 5: Final commit (if any tweaks were needed)**

```bash
git add -A
git commit -m "chore: end-to-end verification fixes for autonomous live call"
```
(Skip if nothing changed.)

---

## Self-Review Notes

- **Spec coverage:** deterministic queue (Task 2), responder (Task 5), QA/Follow-up flaggers (Tasks 3–4), passive note-taker (unchanged, used in Task 7), merged post-call summarizer (Task 6), autonomous WS contract incl. `ai_turn` + auto-TTS + empty-queue silence (Task 7), frontend autonomous mode (Task 8), `flag_type` + `FLAG_TIER_ORDER` (Task 1). Pre-call + context library left untouched, as scoped.
- **Naming consistency:** `LiveCallQueue.enqueue` / `select_next`; `OrchestratorAgent.run(session, flag=...)`; `flag_type` values match `FLAG_TIER_ORDER` exactly; WS message types (`ai_turn`, `tts_audio`, `transcript`, `note`, `coverage_update`) match between Task 7 (backend) and Task 8 (frontend).
- **Tests:** intentionally minimal per hackathon scope — one inline queue sanity check (Task 2) + manual end-to-end run (Task 9).
