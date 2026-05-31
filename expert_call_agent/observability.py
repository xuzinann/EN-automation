"""Optional Weave (W&B) observability.

A no-op when weave is not installed or is explicitly disabled, so the app runs
without a W&B account. Set WEAVE_DISABLED=1 to turn tracing off even when weave
is installed (e.g. when you have no WANDB_API_KEY).
"""
import os

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
    No-ops (returns False) when weave is unavailable/disabled, and never raises —
    a failed init (e.g. no WANDB_API_KEY) just disables tracing instead of
    crashing app startup.
    """
    global _initialized
    if _initialized:
        return True
    if not weave_enabled():
        return False
    project = os.getenv("WEAVE_PROJECT", "http418-expert-call")
    try:
        weave.init(project)
    except Exception as e:  # missing key, network, etc. — degrade, don't crash
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
