from models import AgentAction
import config


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace, for dedup comparisons."""
    return " ".join((text or "").lower().split())


class LiveCallQueue:
    """Deterministic priority queue for live-call flags.

    Flags are ordered by a fixed tier hierarchy (``config.FLAG_TIER_ORDER``);
    ties break FIFO (insertion order). Selected flags are remembered so the
    same question is never surfaced twice. Unselected flags persist across
    turns until they are selected or re-supersede.

    Each flag type may be surfaced at most ``config.LIVE_FLAG_CAPS`` times per
    call (types absent from that map are unlimited). Once a type hits its cap,
    further flags of that type are dropped — the AI raises a contradiction or
    follow-up a bounded number of times, then moves on even if the issue recurs.
    """

    def __init__(self):
        self._pending: list[tuple[int, AgentAction]] = []
        self._asked: set[str] = set()
        self._asked_counts: dict[str, int] = {}
        self._seq = 0

    def _tier_rank(self, action: AgentAction) -> int:
        try:
            return config.FLAG_TIER_ORDER.index(action.flag_type)
        except ValueError:
            return len(config.FLAG_TIER_ORDER)  # unknown types sort last

    def _at_cap(self, flag_type: str) -> bool:
        """True once this flag type has been surfaced its allowed number of times."""
        cap = config.LIVE_FLAG_CAPS.get(flag_type)
        return cap is not None and self._asked_counts.get(flag_type, 0) >= cap

    def enqueue(self, actions: list[AgentAction]) -> None:
        for action in actions:
            key = _normalize(action.content)
            if not key or key in self._asked:
                continue
            if self._at_cap(action.flag_type):
                continue
            if any(_normalize(a.content) == key for _, a in self._pending):
                continue
            self._pending.append((self._seq, action))
            self._seq += 1

    def select_next(self) -> AgentAction | None:
        # Drop any pending flags whose type has hit its per-call cap.
        self._pending = [
            (seq, a) for seq, a in self._pending if not self._at_cap(a.flag_type)
        ]
        if not self._pending:
            return None
        idx = min(
            range(len(self._pending)),
            key=lambda i: (self._tier_rank(self._pending[i][1]), self._pending[i][0]),
        )
        _, action = self._pending.pop(idx)
        self._asked.add(_normalize(action.content))
        self._asked_counts[action.flag_type] = (
            self._asked_counts.get(action.flag_type, 0) + 1
        )
        return action
