# Expert Call Virtual Agent — Technical Architecture

## Overview

A multi-agent system for PE consulting expert calls that handles: project context ingestion, call guide drafting, and a fully autonomous live expert interview with real-time analysis, followed by a post-call synthesis deliverable.

**Target use case:** PE client wants to understand a target company — headcount (N) by role, operational structure, market positioning, etc. — via expert network calls.

**Live-call model:** during the call, the system runs **autonomously**. Specialist agents *flag* issues tied to their role; a **deterministic priority queue** (plain code) selects the single highest-priority flag; an Orchestrator/Responder agent composes the utterance and speaks it to the expert via TTS. Selection is deterministic; only flagging and phrasing use LLMs.

---

## 1. Available Models & Services (Verified in Sandbox)

### Speech-to-Text (STT)

| Method | Status | Notes |
|--------|--------|-------|
| **Google Cloud Speech-to-Text v2 — Chirp 2** | ✅ Shipped (default) | Purpose-built ASR; lower latency + better accuracy than an LLM. `config.STT_PROVIDER="chirp"` |
| Gemini 2.5 Flash (native audio input) | ✅ Fallback | LLM transcription; `config.STT_PROVIDER="gemini"` |

**Shipped:** **Chirp 2** via the Speech-to-Text **v2** regional endpoint, using the implicit recognizer (`recognizers/_`) with inline config and **`autoDecodingConfig`** (the service auto-detects the encoding). Client: `clients/cloud_stt_client.py`; the Gemini fallback is `clients/stt_client.py`. Both share the same `transcribe(bytes, mime_type)` signature, so `routes_call.py` is provider-agnostic. Chirp 2 is **regional** — `config.SPEECH_LOCATION` must be one of `us-central1 | europe-west4 | asia-southeast1`.

**Endpoint:**
```
POST https://{location}-speech.googleapis.com/v2/projects/{PROJECT}/locations/{location}/recognizers/_:recognize
```

**Codec — verified 2026-05-31:** `autoDecodingConfig` decodes the browser's **WebM/Opus** (`MediaRecorder` output, including the live/streaming-header blob) server-side. A genuine Chrome `MediaRecorder` clip transcribes accurately in ~1.6 s, identical to WAV/MP3. **No client- or server-side transcoding (e.g. ffmpeg) is required.**

#### Mic capture — client-side VAD endpointing

The browser performs **utterance endpointing** in JS (no backend change): `MediaRecorder.start()` with **no timeslice** yields one complete WebM blob per utterance on `stop()`. A 50 ms energy poll over a WebAudio `AnalyserNode` (RMS via `getByteTimeDomainData`) cuts the segment after a trailing pause, then restarts. This replaced fixed 5 s windows, whose 2nd+ fragments were headerless and undecodable.

The threshold is **adaptive**: an ambient noise floor is seeded from the first ~400 ms of each mic session, then `threshold = clamp(floor × K, 0.012, 0.045)` (the floor tracks down-fast / up-slow; speech never raises it), with a 2-frame speech-confirm debounce — so a quiet mic and a noisy room both self-calibrate. All knobs are URL-overridable for live tuning without a redeploy (`?vadK`, `?vadSilence`, `?vadFloorMin/Max`, `?vadFrames`, `?vadCalib`, …), and each cut logs `[VAD] send|drop (reason) {ms, peakRms, thr, floor}`. Implemented identically in `static/index.html` and `static/expert.html` — **keep the two in sync.** Real-time gRPC streaming (Chirp 2 `StreamingRecognize`) remains a future upgrade for live partials.

### Text-to-Speech (TTS)

| Method | Status | Notes |
|--------|--------|-------|
| Gemini native audio output | ❌ Not allowlisted | Blocked at project level — needs admin allowlisting |
| **Google Cloud Text-to-Speech API** | ✅ Working | Studio voices available, good quality |

**Recommendation:** Use **Google Cloud TTS API** for voice output.

**Endpoint:**
```
POST https://texttospeech.googleapis.com/v1/text:synthesize
```

**Required header:** `x-goog-user-project: {PROJECT_ID}` (quota project)

**Request format:**
```json
{
  "input": {"text": "Your response text here"},
  "voice": {"languageCode": "en-US", "name": "en-US-Studio-O"},
  "audioConfig": {"audioEncoding": "MP3"}
}
```

**Response:** Base64-encoded audio in `audioContent` field.

### Reasoning Models

| Model | Status | Best For | Latency |
|-------|--------|----------|---------|
| **Gemini 2.5 Pro** | ✅ Working | Complex reasoning, synthesis, long context | Higher |
| **Gemini 2.5 Flash** | ✅ Working | Intent classification, routing, fast Q&A | Low |
| **Gemini 2.0 Flash** | ✅ Working | High-volume simple tasks, streaming | Lowest |
| **Gemini 2.0 Flash Lite** | ✅ Working | Classification, routing, trivial tasks | Lowest |
| **Claude** (via Vertex AI) | ✅ Working | Nuanced reasoning, structured output, conversation | Medium |

**Vertex AI project:** `gp-ct-sbox-sat-gcp0bg-darksoft`
**Region:** `us-central1`

---

## 2. System Architecture

### End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 1: PRE-CALL                        │
│                                                             │
│  Project Context     ──→  Context Ingestion Agent           │
│  (docs, briefs,           (parse, structure, extract        │
│   prior calls)             key questions & targets)         │
│                                ↓                            │
│                        Call Guide Drafter Agent              │
│                        (role-specific questions,             │
│                         N-sizing targets,                   │
│                         areas to probe)                     │
└──────────────────────────────┬──────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────┐
│              PHASE 2: LIVE CALL (autonomous)                │
│                                                             │
│  Expert Audio ─→ VAD ─→ Chirp 2 (STT) ──→ transcript       │
│                            ↓                                │
│         ┌──────────── parallel fan-out ───────────┐         │
│         │  QA Agent        → flag: next question   │         │
│         │  Follow-up Agent → flags: contradiction, │         │
│         │                    probe (+ coverage)    │         │
│         │  Note-taker      → structured notes      │         │
│         │                    (passive, not queued) │         │
│         └────────────────────┬─────────────────────┘         │
│                              ↓                              │
│         DETERMINISTIC PRIORITY QUEUE (plain code)           │
│         order by fixed tier · FIFO within tier · dedup     │
│                              ↓                              │
│         pop top flag → Orchestrator/Responder (Claude)     │
│         (compose natural utterance)                        │
│                              ↓                              │
│                    TTS → audio auto-played to expert        │
└──────────────────────────────┬──────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 3: POST-CALL                       │
│                                                             │
│  Transcript + Notes + Context ──→ Summarizer Agent          │
│                                   (single pass):            │
│                                   • key takeaways           │
│                                     (findings, surprises,   │
│                                      remaining gaps)        │
│                                   • full markdown           │
│                                     deliverable             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Agent Definitions

### Pre-Call Agents

#### Context Ingestion Agent
- **Model:** Claude Sonnet (long context)
- **Input:** Project brief, prior call transcripts, target company info
- **Output:** Structured context object — key entities, hypotheses, known data points, gaps to fill
- **Why:** PE engagements have dense context; this agent builds the knowledge base all other agents reference

#### Call Guide Drafter Agent
- **Model:** Claude (structured reasoning)
- **Input:** Structured context from ingestion
- **Output:** Role-specific interview guide with:
  - Opening questions
  - Deep-dive areas (org structure, N by role, operational metrics)
  - Red-flag probes
  - Specific numbers/data points to confirm

### Live Call Agents

The live call is coordinated by a **deterministic priority queue**, not an LLM. Agents below
either *flag* items into the queue (QA, Follow-up) or run passively (Note-taker). The
Orchestrator consumes the single chosen flag and speaks.

#### Deterministic Priority Queue (not an agent)
- **Type:** Plain code — no model call
- **Role:** Collect all flags emitted in a turn, order them by a **fixed tier hierarchy**, break
  ties FIFO (oldest unaddressed first), drop flags whose question was already asked, and pop the
  single top flag.
- **Tier hierarchy (highest → lowest):**
  1. Contradiction (Follow-up)
  2. Must-ask question (QA)
  3. Follow-up probe (Follow-up)
  4. Should-ask question (QA)
  5. Nice-to-have question (QA)
- **Future extension:** numeric scores within a tier — when two flags genuinely compete (same
  tier), sort by score or hand the tie to the Orchestrator LLM to decide. The `priority` float on
  each action is retained for this.

#### QA Agent (flagger)
- **Model:** Claude (nuanced questioning)
- **Role:** Propose the best next interview question
- **Input:** Call guide + real-time transcript + context
- **Output:** A single flag — the next question, typed `must_ask` / `should_ask` / `nice_to_have`
  from the guide. Does not select or speak; the queue decides whether it runs.

#### Follow-Up Agent (flagger)
- **Model:** Claude Sonnet (nuanced contradiction/probe detection)
- **Role:** Coverage monitor and probe/contradiction generator
- **Input:** Real-time transcript + call guide checklist + known data points
- **Output:**
  - `contradiction` flags (tier 1) when the expert conflicts with prior data
  - `probe` flags (tier 3) when an answer is vague or incomplete
  - Coverage status (% of guide topics addressed) — surfaced to the UI, **not** a queued flag

#### Note Taker Agent (passive)
- **Model:** Claude Haiku (fast, cheap structured output)
- **Role:** Real-time documentation — runs alongside but **never competes in the queue** (it does
  not talk to the expert)
- **Input:** Real-time transcript
- **Output:** Structured notes — timestamped, categorized by topic:
  - Data points (with confidence: stated vs. estimated vs. inferred)
  - Org structure findings
  - N-sizing data per role
  - Quotes worth preserving

#### Orchestrator / Responder
- **Model:** Claude Sonnet (natural live-speech composition)
- **Role:** Composer and voice of the call. Receives the single flag chosen by the deterministic
  queue and turns it into a natural, conversational utterance.
- **Responsibilities:**
  - Phrase the chosen flag smoothly (transitions, acknowledgments)
  - Submit the utterance to the expert via TTS (fully autonomous)
  - Stay silent when the queue is empty (let the expert continue)
- **Note:** Selection of *what* to ask is deterministic (the queue). The Orchestrator decides
  only *how* to say it.

### Post-Call Agent

#### Summarizer (merged)
- **Model:** Claude (synthesis, judgment)
- **Role:** Single post-call synthesis pass — absorbs both the former live Key-Takeaway
  Summarizer and the former Post-Call Summarizer.
- **Input:** Full transcript + accumulated notes + project context
- **Output (one pass):**
  - **Structured key takeaways:** top findings, surprises / deviations from hypotheses,
    remaining gaps
  - **Full markdown deliverable:** executive summary; findings by category; data points
    confirmed (table); gaps remaining; contradictions & red flags; expert credibility
    assessment; recommended follow-up actions

---

## 4. Model Assignment Summary

| Agent | Model | Rationale |
|-------|-------|-----------|
| Context Ingestion | Claude Sonnet | Long context, deep comprehension |
| Call Guide Drafter | Claude | Structured reasoning, consulting domain |
| QA Agent (flagger) | Claude | Nuanced, adaptive questioning |
| Follow-Up Agent (flagger) | Claude Sonnet | Nuanced contradiction/probe detection |
| Note Taker (passive) | Claude Haiku | Fast, cheap structured extraction |
| Orchestrator / Responder | Claude Sonnet | Natural live-speech composition |
| Priority Queue | Deterministic code | Inspectable, testable, zero added latency |
| Post-call Summarizer | Claude | Synthesis, judgment |
| STT | Cloud Speech-to-Text v2 (Chirp 2) | Purpose-built ASR; client-side VAD; Gemini Flash fallback |
| TTS | Google Cloud TTS API (`en-US-Studio-O`) | Studio-quality voice ("Shaun") |

---

## 5. MVP Scope

### Phase 1 MVP
1. **Context ingestion** — upload a project brief, get structured context
2. **Call guide generation** — auto-generate interview guide from context
3. **Live transcription** — Chirp 2 STT with client-side VAD utterance endpointing
4. **Flaggers + deterministic queue** — QA and Follow-up agents flag items; the queue selects
5. **Autonomous responder** — Orchestrator composes the chosen flag and speaks it via TTS
6. **Real-time Note Taker** — passive structured notes during the call
7. **Post-call summary** — merged summarizer produces takeaways + full markdown deliverable

### Phase 2
8. Numeric-score / LLM tiebreak for competing same-tier flags
9. Richer coverage analytics and pacing controls

### Phase 3
10. Integration with telephony (Twilio / equivalent)
11. Persistence (sessions, transcripts, deliverables)
12. Cross-call learning (insights from prior calls inform new ones)

---

## 6. API Authentication

All Vertex AI calls use Application Default Credentials (ADC):

```python
import google.auth
import google.auth.transport.requests

credentials, project = google.auth.default()
auth_req = google.auth.transport.requests.Request()
credentials.refresh(auth_req)

headers = {
    'Authorization': f'Bearer {credentials.token}',
    'Content-Type': 'application/json',
    'x-goog-user-project': project  # required for Cloud TTS
}
```

**Project ID:** `gp-ct-sbox-sat-gcp0bg-darksoft`
**Region:** `us-central1`

> Note: Anthropic-on-Vertex calls use the `:rawPredict` endpoint (see `clients/claude_client.py`),
> not `:generateContent`. Claude runs in region `us-east5`.

---

## 7. APIs to Enable

The following APIs need to be enabled in the GCP project for full functionality:

| API | Current Status | Needed For |
|-----|---------------|------------|
| Vertex AI API | ✅ Enabled | Gemini models |
| Cloud Text-to-Speech API | ✅ Enabled | TTS |
| Cloud Speech-to-Text API (v2) | ✅ Enabled | Primary STT (Chirp 2) — verified working |
| Gemini audio output allowlisting | ❌ Needs admin request | Native Gemini TTS (optional) |

---

## 8. Key Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Latency in the live call | Chirp 2 (fast ASR) + client-side VAD remove fixed windows; flaggers run in parallel; queue selection is instant (no model call); per-call agent timeouts cap stalls |
| Audio output not allowlisted | Use Cloud TTS API (confirmed working) |
| Autonomous TTS speaks something off | Responder phrases only the queue-chosen flag; recent transcript provided for context; monitoring UI retained |
| Token costs with multiple agents | Note-taker on Claude Haiku (cheap); per-type flag caps + queue dedup bound the number of live LLM turns |
| Expert says something contradictory | Follow-Up Agent emits tier-1 contradiction flags that pre-empt other questions |
| Deterministic queue too rigid | Numeric-score / LLM tiebreak designed-in as a future extension (retained `priority` float) |
| Call guide misses key areas | Post-call summarizer identifies remaining gaps for follow-up calls |
