"""pick_watch.PickWatch: count grasps that close on nothing, stop a pick that is going nowhere."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "smolvla"))
from pick_watch import PickWatch  # noqa: E402


def feed(w, positions, dt=1 / 30, hold=10):
    """Each position held for `hold` ticks; returns the first stop reason and the time it came."""
    t = 0.0
    for p in positions:
        for _ in range(hold):
            t += dt
            why = w.update(t, p)
            if why:
                return why, round(t, 2)
    return None, round(t, 2)


def test_clean_grasp_never_counts():
    w = PickWatch()
    assert feed(w, [37, 30, 20, 10, 6, 5, 4.1] + [4.1] * 200) == (None, feed(PickWatch(), [37] * 207)[1])
    assert w.misses == 0


def test_two_misses_then_grasp_is_left_alone():
    w = PickWatch()
    why, _ = feed(w, [37, 3.0, 35, 3.2, 39, 6.0, 6.0, 6.0])
    assert why is None and w.misses == 2


def test_four_seconds_shut_is_left_alone():
    w = PickWatch()
    assert feed(w, [37, 3.0, 37], hold=120)[0] is None


def test_fourth_miss_is_left_alone_fifth_stops():
    w = PickWatch()
    assert feed(w, [37, 3.0, 35, 3.3, 39, 3.1, 41, 3.2, 40])[0] is None and w.misses == 4
    assert feed(w, [3.0])[0] == "misses" and w.misses == 5


def test_staying_closed_counts_once():
    w = PickWatch(stuck_s=0)
    feed(w, [37] + [3.0] * 20)
    assert w.misses == 1


def test_a_close_needs_an_open_first():
    w = PickWatch()  # the pick starts with the jaws reading closed: not a miss, not stuck
    assert feed(w, [3.0] * 30)[0] is None and w.misses == 0


def test_stuck_closed_on_nothing():
    w = PickWatch()
    why, t = feed(w, [37, 3.0], hold=330)  # 11 s open, then shut
    assert why == "stuck_closed" and 16.0 <= t <= 16.1


def test_off_switches():
    w = PickWatch(max_misses=0, stuck_s=0)
    assert feed(w, [37, 3.0, 35, 3.0, 39, 3.0, 41, 3.0] * 3, hold=100)[0] is None
