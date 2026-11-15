"""Notice a pick that is going nowhere, so the loop can go home and try the prompt again.

The VLA picks well on its first try. Once it misses, it tends to keep fumbling where it is. A miss is easy to see on
the gripper: the jaws close on nothing (they bottom out at 2.5-3.7) and reopen. On a brick they stall at 4.1-8.9
(14 Nov traces, 67 picks). Clean picks had at most 2 misses; the default stops at the 5th to leave slow picks room. A pick can also freeze
with the jaws shut on nothing, which never reopens, so a closed-on-nothing stretch of `stuck_s` stops it too.

Pure: no robot, no clock of its own. The loop feeds it (time, gripper position) while the VLA runs the pick.
"""

from __future__ import annotations


class PickWatch:
    def __init__(self, max_misses: int = 5, miss_below: float = 3.8, open_above: float = 12.0, stuck_s: float = 5.0):
        self.max_misses, self.miss_below, self.open_above, self.stuck_s = max_misses, miss_below, open_above, stuck_s
        self.misses = 0
        self._armed = False  # jaws opened since the last miss: the next close on nothing counts
        self._opened = False  # jaws opened at least once in this pick (a closed start reading never counts)
        self._closed_since: float | None = None

    def update(self, t: float, pos: float) -> str | None:
        """Feed one tick. Returns "misses" or "stuck_closed" when the pick should stop, else None."""
        if pos >= self.open_above:
            self._armed = self._opened = True
        empty = pos < self.miss_below
        if empty and self._armed:
            self._armed = False
            self.misses += 1
        if empty and self._opened:
            if self._closed_since is None:
                self._closed_since = t
        else:
            self._closed_since = None
        if self.max_misses > 0 and self.misses >= self.max_misses:
            return "misses"
        if self.stuck_s > 0 and self._closed_since is not None and t - self._closed_since >= self.stuck_s:
            return "stuck_closed"
        return None
