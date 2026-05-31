# Follow-up & Contradiction Suppression Implementation Plan (Revamped)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Project rules: commit straight to `main` (no feature branch); hackathon — no committed test suite, use the lightweight sanity checks given here; run Python as `python3` from inside `expert_call_agent/`.

**Goal:** Stop the live AI from re-raising the *same* contradiction (or follow-up) turn after turn, so its limited interventions are spent on *distinct* concerns — deterministically and without false-merging genuinely different issues.

**Architecture:** Two layers over the existing (already-shipped) hard cap:
1. **C — semantic self-suppression (primary):** The Follow-up agent already receives the rolling transcript, which now includes the *interviewer's own prior turns* (`speaker="interviewer"`, written at `routes_call.py:158`). Instruct it to only surface concerns the interviewer hasn't already raised. Pure prompt change. This is the only layer that understands "same concern, different words."
2. **A — identity-based deterministic dedup (backstop):** The Follow-up agent tags each contradiction with a short canonical `topic`; `LiveCallQueue` suppresses a contradiction whose topic was already raised or is pending. Robust to rewording because it anchors on a stable label, not the prose. On by default with a kill switch.
3. **N-cap (existing, untouched):** `config.LIVE_FLAG_CAPS["contradiction"] = 5` remains the final hard bound on total contradictions surfaced.

**Tech Stack:** FastAPI + WebSockets, Pydantic v2, Gemini Pro (Follow-up agent, `vertexai=True`, `temperature=0.2`), the deterministic `LiveCallQueue`. No new dependencies.

---

## Why this is a revamp (what changed and why)

The original version of this plan proposed "Approach A = fuzzy token-set (Jaccard) dedup on the contradiction text." Two findings killed that approach:

1. **The hardening work already shipped an always-on Jaccard dedup** on flag `content` at `config.LIVE_DEDUP_JACCARD = 0.8` (`live_orchestration.py:67-77`). A second fuzzy mechanism would be redundant.
2. **Token-Jaccard cannot separate a reworded *repeat* from a *distinct* contradiction.** Worked example on the `claim`+`conflicts_with` text:
   - reworded repeats of one contradiction → Jaccard ≈ **0.30–0.50**
   - two genuinely different contradictions → Jaccard ≈ **0.30**

   They overlap, so *no* single threshold both collapses repeats and preserves distinct issues. (The original plan's own verification claimed "expect 1" for its reworded fixtures but actually produces 3 at the 0.6 default.) The shared boilerplate template (`"There may be an inconsistency: … Could you clarify?"`) further muddies content-based similarity.

**Conclusion:** semantic sameness is an LLM judgment (→ Approach **C**), and the deterministic backstop should key on a **stable identity label** the agent assigns (→ Approach **A as topic-identity dedup**), not on fuzzy prose matching. The existing content-Jaccard (0.8) and the per-type cap are left exactly as-is and continue to act as additional backstops.

**Decisions captured (from design discussion 2026-05-31):**
- Approach **C + A**, where **A is now identity/topic-based**, not fuzzy-text-based.
- **Ship C first** (Phase 1, standalone). A is Phase 2.
- **Policy P = 1** (each distinct concern surfaces at most once). C enforces it semantically; A enforces it deterministically per topic.
- Keep `LIVE_FLAG_CAPS` unchanged as the always-on hard backstop. Escalation (P>1, "press once if dodged") remains out of scope.

**Known limitations (document, don't fix here):**
- **C transcript window:** C relies on the prior interviewer turn still being inside `config.MAX_TRANSCRIPT_CONTEXT = 20` entries (≈10 turns). For the 10-turn demo the whole call is in view. In longer calls an issue raised >20 entries ago can fall out of the window — that is exactly what A (topic dedup, which has its own memory) and the cap backstop.
- **A topic collisions:** if the agent labels two genuinely *different* contradictions with the *same* topic, the second is falsely suppressed. Mitigated by instructing the agent to use *specific* topics, by C being primary, and by the kill switch. Tunable later.

**Explicitly out of scope:** embeddings/semantic-vector dedup (no new deps), follow-up (`probe`) topic dedup (probes already get the content-Jaccard + cap; contradictions are the repeat offender), escalation P>1.

---

## File Structure

- `expert_call_agent/agents/followup_agent.py`
  - **C:** extend `system_prompt` + user prompt with the "raise each concern once" rule.
  - **A:** add `"topic"` to the contradiction JSON schema; carry `topic` (+ `claim`/`conflicts_with`) in each contradiction's `metadata`.
- `expert_call_agent/config.py`
  - **A:** add `LIVE_CONTRADICTION_TOPIC_DEDUP = True` (kill switch). Leave `LIVE_FLAG_CAPS` / `LIVE_DEDUP_JACCARD` unchanged.
- `expert_call_agent/live_orchestration.py`
  - **A:** additive topic-identity dedup for contradictions (`_asked_topics`, `_contradiction_topic`, an extra guard in `enqueue`, a record step in `select_next`). **Byte-for-byte unchanged for non-contradiction flags.**

> **Do NOT touch `api/routes_call.py`.** It already calls `queue.tick()` / `enqueue()` / `select_next()` correctly (T8 of the hardening plan). The flag-collection path (`routes_call.py:124-138`) passes contradictions through unchanged; metadata additions ride along automatically.

---

## PHASE 1 — Approach C (primary; ships on its own)

### Task 1: Teach the Follow-up agent to raise each concern only once

**Files:**
- Modify: `expert_call_agent/agents/followup_agent.py:9-22` (`system_prompt`) and `:34` (user prompt)

- [ ] **Step 1: Replace the `system_prompt` (current lines 9-22) with the version below**

This adds the self-suppression rule **and** the `topic` field that Phase 2 needs (harmless in Phase 1 — nothing reads `topic` until Task 3). Everything else is preserved from the current hardened prompt.

```python
    system_prompt = (
        "You monitor an expert call for coverage completeness. "
        "Track which interview guide sections have been addressed. "
        "Flag when the expert gives vague or incomplete answers needing follow-up. "
        "Detect contradictions with known data points.\n\n"
        "RAISE EACH CONCERN ONLY ONCE. The transcript you are given includes the "
        "INTERVIEWER's own earlier turns (speaker 'interviewer'). Before adding a "
        "follow-up or contradiction, check whether the interviewer has ALREADY "
        "raised that same concern earlier in the transcript. If it has, do NOT "
        "raise it again — even if the expert still hasn't answered it satisfactorily, "
        "assume it is noted and move on. Only return follow-ups and contradictions "
        "that are genuinely NEW.\n\n"
        "For each contradiction, also set a short, SPECIFIC `topic` slug naming the "
        "subject of the conflict (e.g. 'revenue', 'gross margin', 'gpu count', "
        "'prior employer'). Reuse the SAME topic wording if that same conflict ever "
        "recurs, and keep topics specific so different conflicts get different topics.\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "coverage": [{"section": "string", "covered_pct": 0.0-1.0, '
        '"answered_questions": ["strings"], "remaining_questions": ["strings"]}],\n'
        '  "followups": [{"question": "string", "reason": "string"}],\n'
        '  "contradictions": [{"topic": "string", "claim": "string", "conflicts_with": "string"}]\n'
        "}\n"
        "No markdown fences or commentary."
    )
```

- [ ] **Step 2: Reinforce in the user prompt (current line 34)**

Replace:

```python
            "Analyze coverage completeness, suggest follow-ups, and flag contradictions."
```

with:

```python
            "Analyze coverage completeness. Suggest follow-ups and flag contradictions, "
            "but ONLY ones not already raised by the interviewer earlier in the transcript above."
```

- [ ] **Step 3: Smoke-import to catch syntax errors**

Run:
```bash
cd /home/kevin/EN-automation/expert_call_agent && python3 -c "from agents.followup_agent import FollowUpAgent; print('import ok')"
```
Expected: `import ok`

- [ ] **Step 4: Live verification (browser demo)**

C is LLM-driven — verify behaviorally, not with a unit test. The server runs on `127.0.0.1:8899` (reachable via the IAP SSH tunnel; it runs without auto-reload, so **restart it** to pick up the change). Drive the UI: **Use Sample Brief → Generate Call Guide → Run Demo** (this pairs the TechServ brief with the CoreWeave demo transcript — the worst case for repeats). Then read the AI-turn tags from the page (selectors verified against `static/index.html:247,727,729`):
```js
() => {
  const cards = [...document.querySelectorAll('#suggestions-list .suggestion-card')]
    .map(c => c.querySelector('.agent-tag')?.textContent.trim());
  const counts = cards.reduce((m,t)=>{m[t]=(m[t]||0)+1;return m;},{});
  return { total: cards.length, counts };
}
```
Expected: the background employer/profile contradiction is raised **roughly once** (allow 1–2 for LLM variance), and the call pivots to `qa_agent — must ask` / other tiers quickly — NOT 5 back-to-back contradictions. The `contradiction:5` cap still guarantees it never exceeds 5 even if C under-suppresses.

- [ ] **Step 5: Commit**

```bash
cd /home/kevin/EN-automation
git add expert_call_agent/agents/followup_agent.py
git commit -m "feat: follow-up agent self-suppresses already-raised concerns (approach C, P=1)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## PHASE 2 — Approach A (identity-based deterministic backstop)

> Phase 2 makes repeat-suppression deterministic even when C slips. It keys on the agent-assigned `topic`, so it is robust to rewording (unlike fuzzy text matching). On by default; flip `LIVE_CONTRADICTION_TOPIC_DEDUP = False` to disable.

### Task 2: Carry contradiction `topic` (+ substance) in metadata

**Files:**
- Modify: `expert_call_agent/agents/followup_agent.py:52-63` (the contradiction-building loop)

> The `topic` JSON field was already added to the schema in Task 1. This task makes the agent's `topic`/`claim`/`conflicts_with` available to the queue via `metadata`.

- [ ] **Step 1: Replace the contradiction loop (current lines 52-63) with:**

```python
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
                metadata={
                    "reason": "contradiction with known data",
                    "topic": contradiction.get("topic", ""),
                    "claim": contradiction.get("claim", ""),
                    "conflicts_with": contradiction.get("conflicts_with", ""),
                },
            ))
```

`content` is unchanged, so the spoken question and the UI rationale (which reads `metadata["reason"]`, `routes_call.py:155`) are unaffected. The extra keys are inert until Task 3.

- [ ] **Step 2: Verify it imports**

```bash
cd /home/kevin/EN-automation/expert_call_agent && python3 -c "from agents.followup_agent import FollowUpAgent; print('import ok')"
```
Expected: `import ok`

### Task 3: Add the topic config switch

**Files:**
- Modify: `expert_call_agent/config.py` (append after the `LIVE_DEDUP_JACCARD` line, currently line 63)

- [ ] **Step 1: Append the switch**

```python

# Contradiction repeat-suppression (identity-based, approach A). A contradiction
# whose canonical `topic` (set by the Follow-up agent) was already raised — or is
# already pending — is dropped, regardless of how its wording varies. This is the
# deterministic backstop to the agent's own semantic self-suppression. Token
# similarity can't tell a reworded repeat from a distinct contradiction, so we key
# on the stable topic label instead. Set False to disable; the per-type cap
# (LIVE_FLAG_CAPS) remains the final hard bound on total contradictions surfaced.
LIVE_CONTRADICTION_TOPIC_DEDUP = True
```

- [ ] **Step 2: Verify it imports**

```bash
cd /home/kevin/EN-automation/expert_call_agent && python3 -c "import config; print(config.LIVE_CONTRADICTION_TOPIC_DEDUP)"
```
Expected: `True`

### Task 4: Topic-identity dedup in `LiveCallQueue`

**Files:**
- Modify: `expert_call_agent/live_orchestration.py` (four small additive edits — `__init__`, a new helper, `enqueue`, `select_next`)

> These edits are additive and gated on `flag_type == "contradiction"`; for every other flag type `_contradiction_topic` returns `""` and the new code is skipped, so non-contradiction behavior is byte-for-byte identical.

- [ ] **Step 1: Add the topic store to `__init__`**

Replace (current lines 49-51):

```python
        self._seq = 0
        self._turn = 0
        self._last_spoke_turn = -(10 ** 9)
```

with:

```python
        self._seq = 0
        self._turn = 0
        self._last_spoke_turn = -(10 ** 9)
        self._asked_topics: set[str] = set()  # contradiction topics already raised
```

- [ ] **Step 2: Add the `_contradiction_topic` helper**

Insert this method immediately after `_at_cap` (i.e., after current line 65, before `_is_duplicate`):

```python
    def _contradiction_topic(self, action: AgentAction) -> str:
        """Normalized canonical topic for a contradiction; '' for any other flag."""
        if action.flag_type != "contradiction" or not action.metadata:
            return ""
        return _normalize(action.metadata.get("topic", ""))
```

- [ ] **Step 3: Guard `enqueue` on topic identity**

Replace the tail of `enqueue` (current lines 86-90):

```python
            key_tokens = _tokens(action.content)
            if self._is_duplicate(key, key_tokens):
                continue
            self._pending.append((self._seq, self._turn, action))
            self._seq += 1
```

with:

```python
            key_tokens = _tokens(action.content)
            if self._is_duplicate(key, key_tokens):
                continue
            if config.LIVE_CONTRADICTION_TOPIC_DEDUP:
                topic = self._contradiction_topic(action)
                if topic and (
                    topic in self._asked_topics
                    or any(self._contradiction_topic(a) == topic for _, _, a in self._pending)
                ):
                    continue  # same contradiction topic already raised or pending
            self._pending.append((self._seq, self._turn, action))
            self._seq += 1
```

- [ ] **Step 4: Record the topic when a contradiction is selected**

Replace the tail of `select_next` (current lines 123-127):

```python
        self._asked_counts[action.flag_type] = (
            self._asked_counts.get(action.flag_type, 0) + 1
        )
        self._last_spoke_turn = self._turn
        return action
```

with:

```python
        self._asked_counts[action.flag_type] = (
            self._asked_counts.get(action.flag_type, 0) + 1
        )
        topic = self._contradiction_topic(action)
        if topic:
            self._asked_topics.add(topic)
        self._last_spoke_turn = self._turn
        return action
```

- [ ] **Step 5: Sanity-check the queue logic (pure, no network)**

Run:
```bash
cd /home/kevin/EN-automation/expert_call_agent && python3 - <<'PY'
import config
from live_orchestration import LiveCallQueue
from models import AgentAction

def contra(topic, claim, conflicts):
    return AgentAction(
        agent_name="followup_agent", action_type="contradiction", flag_type="contradiction",
        content=f"There may be an inconsistency: {claim} vs {conflicts}. Could you clarify?",
        metadata={"reason": "x", "topic": topic, "claim": claim, "conflicts_with": conflicts})

def should(text):
    return AgentAction(agent_name="qa_agent", action_type="suggest_question",
        flag_type="should_ask", content=text)

def run(items):
    q = LiveCallQueue(); n = 0
    for it in items:
        q.tick(); q.enqueue([it])
        if q.select_next():
            n += 1
    return n

# Same topic, different wording each turn -> collapses to 1 (identity-based).
reworded = [
    contra("revenue", "expert says revenue was 1.8B", "brief lists ~500M"),
    contra("revenue", "he reported 1.8 billion run-rate", "sourced near 500 million"),
    contra("revenue", "revenue around 1.8B at exit", "profile suggests 500M"),
]
print("same-topic surfaced:", run(reworded), "(expect 1)")
assert run(reworded) == 1

# Distinct topics -> all survive (bounded by the cap, here well under it).
distinct = [
    contra("revenue", "1.8B", "500M"),
    contra("gross margin", "60 percent", "42 percent"),
    contra("gpu count", "40k H100", "profile says 25k"),
]
print("distinct-topic surfaced:", run(distinct), "(expect 3)")
assert run(distinct) == 3

# Regression: topic dedup must be a no-op for non-contradiction flags.
qs = [should("What is revenue?"), should("How many GPUs?")]
config.LIVE_CONTRADICTION_TOPIC_DEDUP = True
on = run(qs)
config.LIVE_CONTRADICTION_TOPIC_DEDUP = False
off = run(qs)
config.LIVE_CONTRADICTION_TOPIC_DEDUP = True
print("should_ask on/off:", on, off, "(expect equal)")
assert on == off

# Kill switch restores pre-topic behavior for contradictions.
config.LIVE_CONTRADICTION_TOPIC_DEDUP = False
print("switch off, same-topic surfaced:", run(reworded), "(expect 3 — only content-Jaccard/cap apply)")
config.LIVE_CONTRADICTION_TOPIC_DEDUP = True
print("OK")
PY
```
Expected:
```
same-topic surfaced: 1 (expect 1)
distinct-topic surfaced: 3 (expect 3)
should_ask on/off: <n> <n> (expect equal)
switch off, same-topic surfaced: 3 (expect 3 — only content-Jaccard/cap apply)
OK
```
(Note: contradictions are in `LIVE_COOLDOWN_EXEMPT_TIERS`, so the cooldown doesn't suppress them in this harness; `should_ask` may be cooldown-limited, which is why the regression compares on==off rather than asserting a fixed count.)

- [ ] **Step 6: Confirm live behavior (browser)**

Restart the server and re-run the demo as in Task 1 Step 4. With A on, the background contradiction should surface at most once even if C's wording drifts; distinct contradictions still appear. Flip `LIVE_CONTRADICTION_TOPIC_DEDUP=False`, restart, and confirm the call falls back to C + cap behavior (no crash, possibly more repeats).

- [ ] **Step 7: Commit**

```bash
cd /home/kevin/EN-automation
git add expert_call_agent/config.py expert_call_agent/agents/followup_agent.py expert_call_agent/live_orchestration.py
git commit -m "feat: identity-based contradiction dedup by agent-assigned topic (approach A)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Push

After each phase (or at the end), per the shared remote's fetch→rebase→push pattern:
```bash
cd /home/kevin/EN-automation && git fetch origin -q && git rebase origin/main && git push origin main
```

---

## Self-Review (run before handing off)

1. **Spec coverage:** C = Task 1 (prompt). A = Task 2 (metadata `topic`) + Task 3 (switch) + Task 4 (queue topic dedup). P=1 = C never re-emits; A collapses per topic. N-cap = `LIVE_FLAG_CAPS` untouched, still enforced by `_at_cap`. ✅
2. **Default safety / no regression:** For any non-contradiction flag, `_contradiction_topic` returns `""`, so the Task-4 guard is skipped and `enqueue`/`select_next` are identical to today. The Step-5 `on == off` check proves it. ✅
3. **Type consistency:** `_contradiction_topic(action) -> str` is defined in Task 4 Step 2 and used in Steps 3–4; `_asked_topics` defined in Step 1, used in Steps 3–4; `config.LIVE_CONTRADICTION_TOPIC_DEDUP` defined in Task 3, read in Task 4; `metadata["topic"]` written in Task 2, read in Task 4. All consistent. ✅
4. **No placeholders:** every code step is complete; every command has expected output. The Step-5 numbers were hand-computed (reworded topic-Jaccard is irrelevant — identity match is exact). ✅
5. **Known gaps documented:** C transcript-window limit and A topic-collision risk are called out with their backstops. ✅

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-31-followup-contradiction-suppression.md` (this revamp replaces the earlier fuzzy-Jaccard version). Each task is self-contained with complete diffs against the current hardened files. Recommended order: Task 1 (ship C alone) → Tasks 2–4 (A). It does **not** touch `routes_call.py`, so it won't collide with the hardening work.
