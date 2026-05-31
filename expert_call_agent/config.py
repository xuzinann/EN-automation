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
    "qa_agent": ("claude", CLAUDE_MODEL),
    "followup_agent": ("gemini", GEMINI_PRO_MODEL),
    "note_taker": ("gemini", GEMINI_FLASH_MODEL),
    "post_call_summarizer": ("claude", CLAUDE_MODEL),
}

ORCHESTRATOR_TICK_SECONDS = 5
MAX_TRANSCRIPT_CONTEXT = 20

# Live-call flag priority: lower index = higher priority. FIFO breaks ties.
FLAG_TIER_ORDER = ["contradiction", "must_ask", "probe", "should_ask", "nice_to_have"]
