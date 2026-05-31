# Design: Deterministic Live-Call Orchestration + Merged Post-Call Summarizer

**Date:** 2026-05-31
**Status:** Approved (design); implementation pending
**Scope:** Update `expert_call_agent_architecture.md` + write this spec. No code yet.

---

## 1. Motivation

The current live-call phase uses an **LLM Orchestrator** (Gemini Flash) that *reasons* about
which of several pending agent actions to surface. This couples the control flow to a model
call: it adds latency, is non-deterministic, and is hard to inspect or test.

We want the live call to be driven by a **deterministic script** — a priority queue — so that
*selection* of the next action is plain, testable code. LLM intelligence is pushed to the edges:
the flagging agents (which decide *what* is worth raising) and a responder agent (which decides
*how* to phrase the chosen item). We also consolidate the post-call summarization into a single
agent.

## 2. Goals / Non-Goals

**Goals**
- Replace the LLM-based orchestrator-selector with a deterministic priority queue.
- Recast the Orchestrator as a **responder**: it composes and speaks the single chosen flag.
- Keep the Note-taker as a separate, passive real-time documentarian.
- Run the live call **fully autonomously**: the composed utterance auto-plays to the expert via TTS.
- Merge the live Key-Takeaway Summarizer and the Post-Call Summarizer into one post-call agent.

**Non-Goals**
- Numeric priority scoring within a tier (documented as a future extension).
- Human-in-the-loop approval of each turn (autonomous is the chosen mode).
- Telephony integration, persistence, cross-call learning (unchanged from existing roadmap).

## 3. Architecture Overview

Three phases. Pre-call is unchanged. Live call is restructured around a deterministic queue.
Post-call gains the merged summarizer.

```
PHASE 1 — PRE-CALL   (unchanged)
  Brief → Context Ingestion Agent (Claude) → ProjectContext
        → Call Guide Drafter Agent (Claude) → CallGuide

PHASE 2 — LIVE CALL  (deterministic)
  Expert audio → STT (Gemini Flash) → TranscriptEntry(expert)
    → parallel fan-out:
         QA Agent (Claude)        → 0..1 flag   (next guide question)
         Follow-up Agent (Gemini 2.5 Pro) → 0..N flags (contradiction, probe) + coverage status
         Note-taker (Gemini Flash) → structured notes   [recorded, never queued]
    → DETERMINISTIC PRIORITY QUEUE (code): order by fixed tier, FIFO within tier, dedup vs asked
    → pop top flag → Orchestrator/Responder (Gemini Flash): compose natural utterance
    → TTS (Google Cloud) → audio auto-played to expert
    → TranscriptEntry(interviewer); mark flag asked

PHASE 3 — POST-CALL  (merged)
  transcript + notes + context → Summarizer (Claude) →
     structured key takeaways  +  full markdown deliverable
```

## 4. Live-Call Deterministic Orchestration (core)

### 4.1 Components

| Component | Type | Responsibility |
|-----------|------|----------------|
| QA Agent | LLM (Claude) | Emit a flag for the best next guide question, typed by guide priority. |
| Follow-up Agent | LLM (Gemini 2.5 Pro) | Emit contradiction flags + probe flags; produce coverage status (not a flag). |
| Note-taker | LLM (Gemini Flash) | Emit structured notes. Passive — never enqueued. |
| Priority Queue | **Deterministic code** | Collect flags, order by tier, FIFO tiebreak, dedup, pop top. |
| Orchestrator/Responder | LLM (Gemini Flash) | Compose a natural utterance from the single chosen flag. |

### 4.2 Turn cycle

One cycle runs per expert transcript entry:

1. Expert audio → STT → `TranscriptEntry(speaker="expert")`.
2. `asyncio.gather(QA.flag(), Followup.flag(), Notetaker.document())`.
3. Enqueue all QA + Follow-up flags into the priority queue. Notes are recorded directly.
4. Queue orders by fixed tier; within a tier, oldest-unaddressed first; drops any flag whose
   normalized question matches one already asked.
5. Pop the top flag (if any) → Orchestrator/Responder composes the utterance.
6. TTS → audio auto-played; append `TranscriptEntry(speaker="interviewer")`; mark flag asked.
7. **Empty queue → no utterance.** The AI stays silent and lets the expert continue.

### 4.3 Priority hierarchy (fixed tiers)

Highest → lowest:

1. **Contradiction** (Follow-up) — reconcile inconsistencies immediately.
2. **Must-ask question** (QA) — core guide coverage.
3. **Follow-up probe** (Follow-up) — vague/incomplete answer.
4. **Should-ask question** (QA) — standard coverage.
5. **Nice-to-have question** (QA) — only if nothing better is queued.

Within a tier: deterministic **FIFO** (oldest unaddressed flag wins). No numeric scores now.

**Future extension:** attach a numeric score within a tier; when two flags genuinely compete
(same tier), either sort by score or hand the tie to the Orchestrator LLM to decide which to run
with. The `AgentAction.priority` float is retained to make this drop-in later.

### 4.4 Data-model changes (`models.py`)

- `AgentAction` gains `flag_type: str` (one of: `contradiction`, `must_ask`, `probe`,
  `should_ask`, `nice_to_have`). The existing `priority: float` is kept for the future tiebreak.
- Track "asked" flags via a normalized-text set held by the queue/session (no model field needed),
  or add `asked: bool` to `AgentAction` if we keep flags around. **Decision: queue-held set**, to
  avoid mutating stored actions.
- `CoverageStatus` and `StructuredNote` unchanged.

### 4.5 Edge cases

- **No flags:** silence (4.2 step 7).
- **Duplicate question across turns:** dropped by the dedup set (normalized, case-insensitive).
- **Multiple contradictions in one turn:** all tier-1; FIFO picks the oldest, the rest stay queued
  for subsequent turns.
- **Agent error:** a failing flagger is logged and skipped; the queue proceeds with whatever
  flags succeeded (mirrors current `return_exceptions=True` behavior).

## 5. Post-Call Merged Summarizer

A single `Summarizer` (Claude) runs once post-call and returns, in one pass:

- **Structured key takeaways:** top findings, surprises vs. client hypotheses, remaining gaps.
- **Full markdown deliverable:** exec summary; findings by category (headcount/org, ops/tech,
  market/clients, financials); confirmed data points table; gaps; contradictions/red flags;
  expert credibility assessment; recommended follow-ups.

It consumes the full transcript, accumulated notes, and project context. Replaces both the live
`summarizer.py` and the prior `post_call_summarizer.py`.

## 6. Model Assignments (updated)

| Agent | Model | Rationale |
|-------|-------|-----------|
| Context Ingestion | Claude | Long context, comprehension |
| Call Guide Drafter | Claude | Structured reasoning |
| QA Agent | Claude | Nuanced, adaptive questioning |
| Follow-up Agent | Gemini 2.5 Pro | Reasoning + speed |
| Note-taker | Gemini 2.5 Flash | Fast structured extraction |
| Orchestrator/Responder | Gemini 2.5 Flash | Low latency for live speech composition |
| Post-call Summarizer | Claude | Synthesis, judgment |
| STT | Gemini 2.5 Flash | Native audio input |
| TTS | Google Cloud TTS | Only working TTS option |
| Priority Queue | Deterministic code | Inspectable, zero added latency |

## 7. File-by-File Change Plan

| File | Change |
|------|--------|
| `models.py` | Add `flag_type` to `AgentAction`; keep `priority`. |
| `config.py` | Remove `summarizer` from `AGENT_MODELS`; keep `orchestrator` (Gemini Flash, new role); add a `FLAG_TIER_ORDER` constant. |
| `live_orchestration.py` *(new)* | Deterministic priority queue: enqueue, tier-order, FIFO tiebreak, dedup-asked, pop. Pure code, no I/O. |
| `agents/orchestrator.py` | Rewrite selector → responder: input = one flag + recent transcript; output = utterance string. New system prompt for natural, conversational phrasing. |
| `agents/qa_agent.py` | Emit a flag with `flag_type` derived from the guide question's priority. |
| `agents/followup_agent.py` | Emit `contradiction` (tier 1) and `probe` (tier 3) flags; keep coverage status output. |
| `agents/note_taker.py` | Unchanged (passive). |
| `agents/summarizer.py` | **Delete.** |
| `agents/post_call_summarizer.py` | Extend to emit structured takeaways + markdown in one call. |
| `api/routes_call.py` | Rewrite WS loop: parallel flaggers + note-taker → deterministic queue → responder → auto-TTS each turn. Remove LLM-orchestrator selection and the manual `ask_question` approval path. Keep `run_demo`. |
| `api/routes_postcall.py` | Call merged summarizer; return takeaways + summary. |
| `static/index.html` | Autonomous mode: show AI turns, auto-play TTS audio, drop manual approve. Keep transcript/notes/coverage/takeaways panels. |

## 8. Testing Strategy

- **Priority queue (unit, TDD):** tier ordering; FIFO within tier; dedup of asked questions;
  empty-queue → no selection; multiple same-tier flags. Pure code — primary test target.
- **Flag emission (unit, mocked models):** QA emits correct `flag_type` per guide priority;
  Follow-up emits contradiction + probe flags.
- **Responder (unit, mocked):** given one flag, produces a non-empty utterance referencing it.
- **Integration:** run `DEMO_EXPERT_RESPONSES` through the loop with mocked model outputs; assert
  the deterministic selection order matches the tier hierarchy.
- **Post-call:** summarizer returns both structured takeaways and markdown.

## 9. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Autonomous TTS speaks something off | Responder is constrained to phrase only the chosen flag; recent transcript provided for context; monitoring UI retained. |
| Deterministic queue too rigid (misses nuance) | Future numeric-score / LLM-tiebreak extension is designed-in via retained `priority` float. |
| Latency from per-turn TTS + responder | Responder and STT both use Gemini Flash; flaggers run in parallel; queue selection is instant. |
| Dedup misses reworded duplicates | Normalize aggressively; acceptable residual — Follow-up agent also avoids re-probing answered topics. |

## 10. Open Questions

None blocking. The numeric-score tiebreak is deferred by design (Section 4.3).
