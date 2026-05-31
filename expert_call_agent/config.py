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
