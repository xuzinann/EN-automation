# Expert Call Virtual Agent — Technical Architecture

## Overview

A multi-agent system for PE consulting expert calls that handles: project context ingestion, call guide drafting, and live expert interview with real-time analysis.

**Target use case:** PE client wants to understand a target company — headcount (N) by role, operational structure, market positioning, etc. — via expert network calls.

---

## 1. Available Models & Services (Verified in Sandbox)

### Speech-to-Text (STT)

| Method | Status | Notes |
|--------|--------|-------|
| **Gemini 2.5 Pro** (native audio input) | ✅ Working | Best accuracy, handles audio as multimodal input |
| **Gemini 2.5 Flash** (native audio input) | ✅ Working | Slightly faster, near-identical accuracy |
| **Gemini 2.0 Flash** (native audio input) | ✅ Working | Fastest, good for real-time streaming |
| Google Cloud Speech-to-Text API | ❌ Not enabled | Needs API enablement in project |

**Recommendation:** Use **Gemini 2.5 Flash** for STT — best speed/accuracy tradeoff. No separate STT service needed; Gemini handles audio natively as input.

**Vertex AI endpoint:**
```
POST https://us-central1-aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/us-central1/publishers/google/models/gemini-2.5-flash:generateContent
```

**Request format for audio transcription:**
```json
{
  "contents": [{
    "role": "user",
    "parts": [
      {"inlineData": {"mimeType": "audio/mp3", "data": "<base64_audio>"}},
      {"text": "Transcribe this audio exactly."}
    ]
  }]
}
```

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
│                    PHASE 2: LIVE CALL                       │
│                                                             │
│  Expert Audio ──→ Gemini Flash (STT) ──→ Orchestrator      │
│                                              ↓              │
│                    ┌─────────────────────────────────┐      │
│                    │    Parallel Agent Pool           │      │
│                    │                                  │      │
│                    │  ┌──────────────┐  ┌──────────┐ │      │
│                    │  │  QA Agent    │  │ Follow-up│ │      │
│                    │  │  (drive the  │  │  Agent   │ │      │
│                    │  │   interview) │  │ (monitor │ │      │
│                    │  └──────────────┘  │  coverage│ │      │
│                    │                     │  & probe)│ │      │
│                    │  ┌──────────────┐  └──────────┘ │      │
│                    │  │ Note Taker   │               │      │
│                    │  │ (structured  │  ┌──────────┐ │      │
│                    │  │  real-time   │  │ Key      │ │      │
│                    │  │  notes)      │  │ Takeaway │ │      │
│                    │  └──────────────┘  │Summarizer│ │      │
│                    │                     └──────────┘ │      │
│                    └─────────────────────────────────┘      │
│                              ↓                              │
│                    Orchestrator → TTS → Audio to caller     │
└──────────────────────────────┬──────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 3: POST-CALL                       │
│                                                             │
│  Note Taker output ──→ Final Summary Agent                  │
│  Key Takeaways     ──→ (structured deliverable,             │
│  Call transcript        findings vs. gaps,                  │
│                         next steps)                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Agent Definitions

### Pre-Call Agents

#### Context Ingestion Agent
- **Model:** Claude or Gemini 2.5 Pro (long context)
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

#### Orchestrator (Moderator)
- **Model:** Gemini 2.5 Flash (low latency)
- **Role:** Traffic controller — decides which agent output to surface, manages turn-taking
- **Responsibilities:**
  - Routes transcribed audio to all agents simultaneously
  - Selects next question/action from QA or Follow-up agent
  - Prevents agents from talking over each other
  - Monitors call time and pacing

#### QA Agent
- **Model:** Claude (nuanced questioning)
- **Role:** Primary interviewer — drives the conversation
- **Input:** Call guide + real-time transcript + context
- **Output:** Next question to ask, adapted based on expert's responses
- **Behavior:** Follows the call guide but adapts dynamically

#### Follow-Up Agent
- **Model:** Gemini 2.5 Pro (reasoning + speed)
- **Role:** Coverage monitor and probe generator
- **Input:** Real-time transcript + call guide checklist
- **Output:**
  - Coverage tracker (% of guide topics addressed)
  - Follow-up probes when expert gives vague or incomplete answers
  - Flags contradictions with prior data
- **Behavior:** Suggests follow-ups to the Orchestrator, does not speak directly

#### Note Taker Agent
- **Model:** Gemini 2.5 Flash (fast, structured output)
- **Role:** Real-time documentation
- **Input:** Real-time transcript
- **Output:** Structured notes — timestamped, categorized by topic:
  - Data points (with confidence: stated vs. estimated vs. implied)
  - Org structure findings
  - N-sizing data per role
  - Quotes worth preserving

#### Key Takeaway Summarizer
- **Model:** Claude (synthesis)
- **Role:** Running summary of critical insights
- **Input:** Real-time transcript + Note Taker output
- **Output:**
  - Top findings so far (updated every few minutes)
  - Surprises / deviations from hypothesis
  - Remaining gaps

---

## 4. Model Assignment Summary

| Agent | Model | Rationale |
|-------|-------|-----------|
| Context Ingestion | Claude / Gemini 2.5 Pro | Long context, deep comprehension |
| Call Guide Drafter | Claude | Structured reasoning, consulting domain |
| Orchestrator | Gemini 2.5 Flash | Low latency, routing decisions |
| QA Agent | Claude | Nuanced, adaptive questioning |
| Follow-Up Agent | Gemini 2.5 Pro | Reasoning + faster than Claude |
| Note Taker | Gemini 2.5 Flash | Speed, structured extraction |
| Key Takeaway Summarizer | Claude | Synthesis, judgment |
| STT | Gemini 2.5 Flash | Native audio input, fast |
| TTS | Google Cloud TTS API | Only working TTS option |

---

## 5. MVP Scope

### Phase 1 MVP (Recommended starting point)
1. **Context ingestion** — upload a project brief, get structured context
2. **Call guide generation** — auto-generate interview guide from context
3. **Live transcription** — Gemini STT during the call
4. **Single QA agent** — suggest next questions based on transcript + guide
5. **Post-call summary** — structured notes + key takeaways from transcript

### Phase 2
6. Add Follow-Up Agent for coverage monitoring
7. Add real-time Note Taker
8. Add TTS for voice output (automated caller)

### Phase 3
9. Full multi-agent orchestration during live calls
10. Integration with telephony (Twilio / equivalent)
11. Cross-call learning (insights from prior calls inform new ones)

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

---

## 7. APIs to Enable

The following APIs need to be enabled in the GCP project for full functionality:

| API | Current Status | Needed For |
|-----|---------------|------------|
| Vertex AI API | ✅ Enabled | Gemini models |
| Cloud Text-to-Speech API | ✅ Enabled | TTS |
| Cloud Speech-to-Text API | ❌ Needs enabling | Fallback STT (optional) |
| Gemini audio output allowlisting | ❌ Needs admin request | Native Gemini TTS (optional) |

---

## 8. Key Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Latency in multi-agent live call | Use Flash for latency-sensitive agents; run agents in parallel, not sequentially |
| Audio output not allowlisted | Use Cloud TTS API (confirmed working) |
| Token costs with multiple agents | Use Flash/Lite for simple tasks; reserve Pro/Claude for reasoning |
| Expert says something contradictory | Follow-Up Agent specifically monitors for contradictions |
| Call guide misses key areas | Post-call Completeness Critic identifies gaps for follow-up calls |
