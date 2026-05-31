from models import AgentAction
import config


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace, for dedup comparisons."""
    return " ".join((text or "").lower().split())


def _tokens(text: str) -> set[str]:
    return set(_normalize(text).split())


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


class LiveCallQueue:
    """Deterministic priority queue for live-call flags.

    Flags are ordered by a fixed tier hierarchy (``config.FLAG_TIER_ORDER``);
    ties break FIFO (insertion order). Selection is fully deterministic given a
    fixed set of input flags.

    Beyond tiering, the queue enforces four live-call policies:

    * **Dedup** — exact and near-duplicate (token-set Jaccard >=
      ``config.LIVE_DEDUP_JACCARD``) flags are never queued twice, whether
      against already-asked or still-pending flags.
    * **Per-type caps** — each flag type may be surfaced at most
      ``config.LIVE_FLAG_CAPS`` times per call (absent = unlimited).
    * **TTL** — a pending flag not surfaced within
      ``config.LIVE_FLAG_TTL_TURNS`` expert turns is dropped as stale.
    * **Cooldown** — after the AI speaks it stays quiet for
      ``config.LIVE_MIN_TURNS_BETWEEN_QUESTIONS`` turns, except for urgent tiers
      in ``config.LIVE_COOLDOWN_EXEMPT_TIERS``.

    Call ``tick()`` once per expert turn, before ``enqueue``/``select_next``.
    """

    def __init__(self):
        self._pending: list[tuple[int, int, AgentAction]] = []  # (seq, turn, action)
        self._asked: set[str] = set()
        self._asked_tokens: list[set[str]] = []
        self._asked_counts: dict[str, int] = {}
        self._seq = 0
        self._turn = 0
        self._last_spoke_turn = -(10 ** 9)

    def tick(self) -> None:
        """Advance the turn counter; call once per processed expert entry."""
        self._turn += 1

    def _tier_rank(self, action: AgentAction) -> int:
        try:
            return config.FLAG_TIER_ORDER.index(action.flag_type)
        except ValueError:
            return len(config.FLAG_TIER_ORDER)  # unknown types sort last

    def _at_cap(self, flag_type: str) -> bool:
        cap = config.LIVE_FLAG_CAPS.get(flag_type)
        return cap is not None and self._asked_counts.get(flag_type, 0) >= cap

    def _is_duplicate(self, key: str, key_tokens: set[str]) -> bool:
        if key in self._asked:
            return True
        for toks in self._asked_tokens:
            if _jaccard(key_tokens, toks) >= config.LIVE_DEDUP_JACCARD:
                return True
        for _, _, a in self._pending:
            other = _normalize(a.content)
            if other == key or _jaccard(key_tokens, _tokens(a.content)) >= config.LIVE_DEDUP_JACCARD:
                return True
        return False

    def enqueue(self, actions: list[AgentAction]) -> None:
        for action in actions:
            key = _normalize(action.content)
            if not key:
                continue
            if self._at_cap(action.flag_type):
                continue
            key_tokens = _tokens(action.content)
            if self._is_duplicate(key, key_tokens):
                continue
            self._pending.append((self._seq, self._turn, action))
            self._seq += 1

    def select_next(self) -> AgentAction | None:
        # Drop flags that are capped or stale (TTL exceeded).
        self._pending = [
            (seq, turn, a)
            for seq, turn, a in self._pending
            if not self._at_cap(a.flag_type)
            and (self._turn - turn) <= config.LIVE_FLAG_TTL_TURNS
        ]
        if not self._pending:
            return None

        in_cooldown = (
            self._turn - self._last_spoke_turn
        ) < config.LIVE_MIN_TURNS_BETWEEN_QUESTIONS

        candidates = [
            (seq, turn, a)
            for seq, turn, a in self._pending
            if not in_cooldown or a.flag_type in config.LIVE_COOLDOWN_EXEMPT_TIERS
        ]
        if not candidates:
            return None

        seq, turn, action = min(
            candidates, key=lambda c: (self._tier_rank(c[2]), c[0])
        )
        self._pending.remove((seq, turn, action))

        key = _normalize(action.content)
        self._asked.add(key)
        self._asked_tokens.append(_tokens(action.content))
        self._asked_counts[action.flag_type] = (
            self._asked_counts.get(action.flag_type, 0) + 1
        )
        self._last_spoke_turn = self._turn
        return action
