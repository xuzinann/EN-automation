"""Optional Weave (W&B) observability.

A no-op when weave is not installed or is explicitly disabled, so the app runs
without a W&B account. Set WEAVE_DISABLED=1 to turn tracing off even when weave
is installed (e.g. when you have no WANDB_API_KEY).
"""
import os
from pathlib import Path

# Load expert_call_agent/.env (if python-dotenv is available) so WANDB_API_KEY /
# WEAVE_PROJECT can live in a gitignored .env instead of the shell. Explicit path
# keeps it cwd-independent, and load_dotenv does NOT override vars already set in
# the environment (so an inline WANDB_API_KEY=... still wins). Best-effort: any
# failure just means env must come from the shell.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except Exception:
    pass

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
    Never blocks and never raises: tracing requires an explicit WANDB_API_KEY.
    Without one, wandb/weave would fall back to an INTERACTIVE login prompt that
    reads stdin and hangs a headless server (a hang a try/except can't catch), so
    we skip init entirely. On no key, or any init failure, tracing is just
    disabled rather than prompting or crashing.
    """
    global _initialized
    if _initialized:
        return True
    if not weave_enabled():
        return False
    if not os.getenv("WANDB_API_KEY"):
        # No key → do NOT call weave.init: it would drop into wandb's interactive
        # credential prompt and stall app startup. Degrade to no tracing instead.
        print("[observability] WANDB_API_KEY not set; Weave tracing disabled")
        return False
    project = os.getenv("WEAVE_PROJECT", "http418-expert-call")
    try:
        weave.init(project)
    except Exception as e:  # invalid key, network, etc. — degrade, don't crash
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
