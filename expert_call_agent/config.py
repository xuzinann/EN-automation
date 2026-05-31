GCP_PROJECT = "gp-ct-sbox-sat-gcp0bg-darksoft"

GEMINI_REGION = "us-central1"
CLAUDE_REGION = "us-east5"

GEMINI_PRO_MODEL = "gemini-2.5-pro"
GEMINI_FLASH_MODEL = "gemini-2.5-flash"
CLAUDE_MODEL = "claude-sonnet-4@20250514"
CLAUDE_HAIKU_MODEL = "claude-haiku-4-5@20251001"

ANTHROPIC_VERSION = "vertex-2023-10-16"

TTS_ENDPOINT = "https://texttospeech.googleapis.com/v1/text:synthesize"
TTS_VOICE = "en-US-Studio-O"
TTS_LANGUAGE = "en-US"

# Speech-to-Text provider. "chirp" = Google Cloud Speech-to-Text v2 Chirp 2 (a
# purpose-built ASR model — lower latency, better accuracy than an LLM). "gemini"
# = transcribe via Gemini Flash (the LLM fallback). chirp_2 is REGIONAL, so
# SPEECH_LOCATION must be one of: us-central1 | europe-west4 | asia-southeast1.
STT_PROVIDER = "chirp"
SPEECH_LOCATION = "us-central1"
SPEECH_MODEL = "chirp_2"
SPEECH_LANGUAGE_CODES = ["en-US"]

# Per-agent (provider, model). BaseAgent._call_model / _call_model_text branch on
# provider, and each agent's model is honored per call (ClaudeClient.generate(model=…)
# / GeminiClient.generate(model=…)), so a swap here needs no other code edits.
# All live + structured agents now run on Claude; only STT still uses Gemini Flash.
AGENT_MODELS = {
    "context_ingestion": ("claude", CLAUDE_MODEL),
    "call_guide_drafter": ("claude", CLAUDE_MODEL),
    "orchestrator": ("claude", CLAUDE_MODEL),
    "qa_agent": ("claude", CLAUDE_MODEL),
    "followup_agent": ("claude", CLAUDE_MODEL),
    "note_taker": ("claude", CLAUDE_HAIKU_MODEL),
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

# Contradiction repeat-suppression (identity-based, approach A). A contradiction
# whose canonical `topic` (set by the Follow-up agent) was already raised — or is
# already pending — is dropped, regardless of how its wording varies. This is the
# deterministic backstop to the agent's own semantic self-suppression. Token
# similarity can't tell a reworded repeat from a distinct contradiction, so we key
# on the stable topic label instead. Set False to disable; the per-type cap
# (LIVE_FLAG_CAPS) remains the final hard bound on total contradictions surfaced.
LIVE_CONTRADICTION_TOPIC_DEDUP = True

# Seconds to wait after the expert joins a live call before the agent speaks its
# opening turn — lets the page render, the audio context unlock, and the mic spin
# up so the greeting isn't clipped. Set to 0 to disable the delay.
LIVE_OPENING_DELAY_SECONDS = 2.0
