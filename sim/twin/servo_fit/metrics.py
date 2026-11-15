"""Servo response metrics of the SO-101 follower from (command, reading) sequences, shared by the real data and the
twin replays (plain numpy, no Isaac).

Indexing (as sim/twin/dynamics/data.py): cmd[t] is what the controller wrote on tick t (before LeRobot's
max_relative_target clamp), meas[t] the reading taken on tick t before the write. The servo goal of tick t is
g[t] = clip(cmd[t], meas[t] - 10, meas[t] + 10); it shows first in meas[t + 1] (delay 0 in this indexing).

Per joint (LeRobot deg, gripper 0-100 units):
* fo_fit: first-order + dead-time + rate-limit model y[k+1] = y[k] + clip(K (g[k-d] - y[k]), +-v), fitted by 20-tick
  open-loop simulation error from the real reading at each window start (windows where the goal moves >= 2 deg).
  tau = -dt / ln(1 - K).
* lag: velocity cross-correlation lag (ticks) between dg and dmeas.
* vmax: p99 |dmeas| per tick on ticks with |g - meas| >= 8 (the servo is saturated).
* stops: the goal moves >= 3 deg in <= 10 ticks, then holds (range <= 0.3) >= 20 ticks: overshoot past the held
  goal (in the move direction, deg), the error at the end of the hold (signed, + = short of the goal), and the
  10-90 % rise time of the reading.
* small: steps of 1-5 deg (<= 3 ticks) out of a >= 10-tick hold, followed by a >= 15-tick hold (the residual's
  scale): achieved fraction after 15 ticks, ticks to the first count change and to 63 % of the step.
* gripper closes: the command falls from >= 25 to < 15; ticks from the command onset to the first reading change,
  to reading < 15 and to settled (within 0.3 of the final value for 5 ticks); final value -> empty (< 4.5) or brick
  (4.5-8). Opens: the command rises from < 15 to >= 25; ticks to reading >= 25.
"""
from __future__ import annotations

import numpy as np

DT = 1.0 / 30.0
CLAMP = 10.0
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def goal(cmd: np.ndarray, meas: np.ndarray, clamp: float = CLAMP) -> np.ndarray:
    return np.clip(cmd, meas - clamp, meas + clamp)


# ---------------------------------------------------------------------------------------------- first-order fit
def _windows(g, y, H, dmax, move=2.0, stride=5, move_hi=None):
    s = []
    for a in range(dmax, len(y) - H - 1, stride):
        r = np.ptp(g[a - dmax:a + H])
        if r >= move and (move_hi is None or r <= move_hi):
            s.append(a)
    return np.array(s, int)


def fo_sim(g, y0, K, v, d, starts, H):
    """Batched open-loop sim over windows: returns (W, H) predicted readings y[a+1..a+H]."""
    W = len(starts)
    y = np.full(W, 0.0) + y0
    out = np.zeros((W, H))
    for k in range(H):
        e = g[starts + k - d] - y
        y = y + np.clip(K * e, -v, v)
        out[:, k] = y
    return out


KS = np.arange(0.05, 0.96, 0.025)
VS = (2, 3, 4, 5, 6, 7, 8, 10, 1e9)


def fo_fit(cmd_list, meas_list, j, clamp=CLAMP, H=20, delays=(0, 1, 2), Ks=KS, vs=VS, move=2.0, move_hi=None):
    """move / move_hi: goal range per window (deg); move=1, move_hi=5 is the small-correction (residual-scale) fit."""
    G, Y, S = [], [], []
    for c, m in zip(cmd_list, meas_list, strict=True):
        g = goal(c[:, j], m[:, j], clamp)
        st = _windows(g, m[:, j], H, max(delays), move=move, move_hi=move_hi)
        if len(st):
            G.append(g)
            Y.append(m[:, j])
            S.append(st)
    if not S:
        return None
    best = None
    res = {}
    for d in delays:
        bd = None
        for v in vs:
            for K in Ks:
                se, n = 0.0, 0
                for g, y, st in zip(G, Y, S, strict=True):
                    p = fo_sim(g, y[st], K, v, d, st, H)
                    r = y[st[:, None] + 1 + np.arange(H)[None]]
                    se += float(((p - r) ** 2).sum())
                    n += p.size
                rm = (se / n) ** 0.5
                if bd is None or rm < bd[0]:
                    bd = (rm, K, v)
        res[f"d{d}"] = {"rmse": round(bd[0], 4), "K": round(float(bd[1]), 3), "v": bd[2]}
        if best is None or bd[0] < best[0]:
            best = (bd[0], d, bd[1], bd[2])
    rm, d, K, v = best
    return {"delay_ticks": int(d), "K": round(float(K), 3), "tau_s": round(float(-DT / np.log(1 - K)), 4),
            "v_per_tick": (None if v > 1e8 else v), "rmse": round(rm, 4), "by_delay": res,
            "n_windows": int(sum(len(s) for s in S))}


# ---------------------------------------------------------------------------------------------- simple stats
def lag_xcorr(cmd_list, meas_list, j, clamp=CLAMP, lags=(0, 1, 2, 3, 4, 5)):
    num = np.zeros(len(lags))
    for c, m in zip(cmd_list, meas_list, strict=True):
        g = goal(c[:, j], m[:, j], clamp)
        dg, dm = np.diff(g), np.diff(m[:, j])
        for i, L in enumerate(lags):
            if len(dg) > L + 2:
                num[i] += float((dg[:len(dg) - L] * dm[L:]).sum())
    return int(list(lags)[int(np.argmax(num))]), [round(float(x / max(num.max(), 1e-9)), 3) for x in num]


def vmax(cmd_list, meas_list, j, clamp=CLAMP, sat=8.0):
    v = []
    for c, m in zip(cmd_list, meas_list, strict=True):
        g = goal(c[:, j], m[:, j], clamp)
        e = (g - m[:, j])[:-1]
        dm = np.diff(m[:, j])
        k = np.abs(e) >= sat
        v.append(np.abs(dm[k]))
    v = np.concatenate(v) if v else np.zeros(0)
    if len(v) < 20:
        return None
    return {"p50": round(float(np.percentile(v, 50)), 2), "p90": round(float(np.percentile(v, 90)), 2),
            "p99": round(float(np.percentile(v, 99)), 2), "n": int(len(v))}


def _holds_after(g, a, n, tol):
    seg = g[a:a + n]
    return len(seg) == n and np.ptp(seg) <= tol


def stops(cmd_list, meas_list, j, move=3.0, hold=20, tol=0.3):
    ov, err, rise = [], [], []
    for c, m in zip(cmd_list, meas_list, strict=True):
        cj, y = c[:, j], m[:, j]
        t = 10
        while t < len(cj) - hold - 1:
            if _holds_after(cj, t, hold, tol) and not _holds_after(cj, t - 1, hold, tol):
                a = t - 10
                dmov = cj[t] - cj[a]
                if abs(dmov) >= move:
                    s = np.sign(dmov)
                    yy = y[t:t + hold]
                    ov.append(float(((yy - cj[t]) * s).max()))
                    err.append(float((cj[t] - y[t + hold - 1]) * s))
                    y0, y1 = y[a], y[t + hold - 1]
                    if abs(y1 - y0) > 1.0:
                        f = (y[a:t + hold] - y0) / (y1 - y0)
                        i10, i90 = int(np.argmax(f >= 0.1)), int(np.argmax(f >= 0.9))
                        rise.append(i90 - i10)
                    t += hold
                    continue
            t += 1
    if not ov:
        return None
    ov, err = np.array(ov), np.array(err)
    return {"n": int(len(ov)), "overshoot_p50": round(float(np.median(ov)), 3),
            "overshoot_p90": round(float(np.percentile(ov, 90)), 3),
            "frac_overshoot_gt_0p3": round(float((ov > 0.3).mean()), 3),
            "end_err_p50": round(float(np.median(err)), 3), "end_abs_err_p50": round(float(np.median(np.abs(err))), 3),
            "rise_10_90_ticks_p50": (round(float(np.median(rise)), 1) if rise else None)}


def small_steps(cmd_list, meas_list, j, lo=1.0, hi=5.0, pre=8, post=12, tol=0.35):
    rows = []
    for c, m in zip(cmd_list, meas_list, strict=True):
        cj, y = c[:, j], m[:, j]
        for t in range(pre, len(cj) - post - 4):
            if not _holds_after(cj, t - pre, pre, tol):
                continue
            for w in (1, 2, 3):
                if _holds_after(cj, t + w - 1, post, tol) and not _holds_after(cj, t - pre, pre + 1, tol):
                    A = cj[t + w - 1] - cj[t - 1]
                    if lo <= abs(A) <= hi:
                        s = np.sign(A)
                        r = (y[t - 1:t + w + post - 1] - y[t - 1]) * s
                        moved = np.nonzero(np.abs(r) >= 0.08)[0]
                        reach = np.nonzero(r >= 0.63 * abs(A))[0]
                        rows.append((abs(A), r[-1] / abs(A), moved[0] if len(moved) else np.nan,
                                     reach[0] if len(reach) else np.nan))
                    break
    if not rows:
        return None
    R = np.array(rows)
    out = {}
    for a, b in ((1.0, 2.0), (2.0, 3.5), (3.5, 5.0)):
        k = (R[:, 0] >= a) & (R[:, 0] <= b)
        if k.sum() < 2:
            continue
        out[f"{a:g}-{b:g}"] = {"n": int(k.sum()), "achieved_frac_p50": round(float(np.median(R[k, 1])), 3),
                               "first_move_ticks_p50": round(float(np.nanmedian(R[k, 2])), 1),
                               "t63_ticks_p50": (round(float(np.nanmedian(R[k, 3])), 1)
                                                 if np.isfinite(R[k, 3]).any() else None),
                               "frac_reached_63": round(float(np.isfinite(R[k, 3]).mean()), 3)}
    return out


# ---------------------------------------------------------------------------------------------- gripper
def gripper_events(cmd, meas, open_hi=25.0, close_lo=15.0, win=45):
    gc, gm = cmd[:, 5], meas[:, 5]
    ev = []
    t = 1
    while t < len(gc) - 5:
        if gc[t - 1] >= open_hi and gc[t] < gc[t - 1]:
            # follow the falling command to below close_lo within 15 ticks
            k = t
            while k < len(gc) - 1 and k - t < 15 and gc[k] >= close_lo:
                k += 1
            if gc[k] < close_lo:
                on = t - 1  # walk back to the start of the falling run; onset = its first lowered command
                while on > 0 and gc[on - 1] > gc[on] + 0.3:
                    on -= 1
                on += 1  # meas[on + 1] is the first reading that can respond (t_first = 1 is no lag)
                end = k
                while end < min(len(gc) - 1, k + win) and gc[end + 1] < close_lo:
                    end += 1
                y = gm[on:end + 1]
                m0 = gm[on]
                if gc[on - 1] < open_hi:
                    t = end + 1
                    continue
                if len(y) < 20 or m0 < 20:
                    t = end + 1
                    continue
                fin = float(np.median(gm[max(on, end - 5):end + 1]))
                ch = np.nonzero(y < m0 - 1.0)[0]
                b15 = np.nonzero(y < close_lo)[0]
                ok = np.abs(y - fin) <= 0.3
                st = None
                for i in range(len(y) - 4):
                    if ok[i:i + 5].all():
                        st = i
                        break
                cs = np.nonzero(gc[on:end + 1] < close_lo)[0]
                dv = -np.diff(y)
                ev.append({"on": int(on), "end": int(end), "m0": round(float(m0), 2), "final": round(fin, 2),
                           "cmd_final": round(float(gc[end]), 2),
                           "kind": "empty" if fin < 4.5 else ("brick" if fin <= 8.0 else "other"),
                           "t_first": (int(ch[0]) if len(ch) else None),
                           "t_lt15": (int(b15[0]) if len(b15) else None),
                           "t_settled": (int(st) if st is not None else None),
                           "t_cmd_lt15": (int(cs[0]) if len(cs) else None),
                           "peak_close_speed": round(float(dv.max()), 2) if len(dv) else None})
                t = end + 1
                continue
        t += 1
    return ev


def gripper_opens(cmd, meas, open_hi=25.0, close_lo=15.0, win=30):
    gc, gm = cmd[:, 5], meas[:, 5]
    ev = []
    t = 1
    while t < len(gc) - win:
        if gc[t - 1] < close_lo and gc[t] > gc[t - 1] + 0.3 and gm[t - 1] < close_lo:
            k = t
            while k < len(gc) - 1 and k - t < 15 and gc[k] < open_hi:
                k += 1
            if gc[k] >= open_hi:
                y = gm[t - 1:t - 1 + win]
                ch = np.nonzero(y > y[0] + 1.0)[0]
                a = np.nonzero(y >= open_hi)[0]
                ev.append({"on": int(t - 1), "t_first": int(ch[0]) if len(ch) else None,
                           "t_gt25": int(a[0]) if len(a) else None,
                           "peak_open_speed": round(float(np.diff(y).max()), 2)})
                t = k + win
                continue
        t += 1
    return ev


def summarize_gripper(ev):
    out = {}
    for kind in ("empty", "brick", "other"):
        e = [x for x in ev if x["kind"] == kind]
        if not e:
            continue

        def med(key, e=e):
            v = [x[key] for x in e if x[key] is not None]
            return round(float(np.median(v)), 2) if v else None

        def pct(key, q, e=e):
            v = [x[key] for x in e if x[key] is not None]
            return round(float(np.percentile(v, q)), 2) if v else None

        out[kind] = {"n": len(e), "final_p50": med("final"), "t_first_p50": med("t_first"),
                     "t_lt15_p50": med("t_lt15"), "t_lt15_p10_p90": [pct("t_lt15", 10), pct("t_lt15", 90)],
                     "t_settled_p50": med("t_settled"), "t_settled_p10_p90": [pct("t_settled", 10),
                                                                              pct("t_settled", 90)],
                     "t_cmd_lt15_p50": med("t_cmd_lt15"), "peak_close_speed_p50": med("peak_close_speed"),
                     "m0_p50": med("m0")}
    return out


def all_metrics(cmd_list, meas_list, fit=True, joints=(0, 1, 2, 3, 4, 5), clamp=CLAMP):
    """clamp: LeRobot max_relative_target of the recording (10 in the deployed controller; teleop ran without)."""
    out = {}
    for j in joints:
        jn = JOINTS[j]
        r = {"lag_xcorr": lag_xcorr(cmd_list, meas_list, j, clamp), "vmax": vmax(cmd_list, meas_list, j, clamp)}
        if j < 5:
            r["stops"] = stops(cmd_list, meas_list, j)
            r["small"] = small_steps(cmd_list, meas_list, j)
        if fit:
            r["fo_fit"] = fo_fit(cmd_list, meas_list, j, clamp)
        out[jn] = r
    ev = []
    for c, m in zip(cmd_list, meas_list, strict=True):
        ev += gripper_events(c, m)
    out["gripper_close"] = summarize_gripper(ev)
    op = []
    for c, m in zip(cmd_list, meas_list, strict=True):
        op += gripper_opens(c, m)
    if op:
        out["gripper_open"] = {"n": len(op),
                               "t_first_p50": float(np.median([x["t_first"] for x in op if x["t_first"] is not None])),
                               "t_gt25_p50": float(np.median([x["t_gt25"] for x in op if x["t_gt25"] is not None]))
                               if any(x["t_gt25"] is not None for x in op) else None,
                               "peak_open_speed_p50": float(np.median([x["peak_open_speed"] for x in op]))}
    return out
