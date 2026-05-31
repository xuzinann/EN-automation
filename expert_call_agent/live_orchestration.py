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
    """

    def __init__(self):
        self._pending: list[tuple[int, AgentAction]] = []
        self._asked: set[str] = set()
        self._seq = 0

    def _tier_rank(self, action: AgentAction) -> int:
        try:
            return config.FLAG_TIER_ORDER.index(action.flag_type)
        except ValueError:
            return len(config.FLAG_TIER_ORDER)  # unknown types sort last

    def enqueue(self, actions: list[AgentAction]) -> None:
        for action in actions:
            key = _normalize(action.content)
            if not key or key in self._asked:
                continue
            if any(_normalize(a.content) == key for _, a in self._pending):
                continue
            self._pending.append((self._seq, action))
            self._seq += 1

    def select_next(self) -> AgentAction | None:
        if not self._pending:
            return None
        idx = min(
            range(len(self._pending)),
            key=lambda i: (self._tier_rank(self._pending[i][1]), self._pending[i][0]),
        )
        _, action = self._pending.pop(idx)
        self._asked.add(_normalize(action.content))
        return action
