"""Held-out overlays: real frame (real mask tinted blue), sim outline with joe_cal2 (red) and the refit C (green).

    python sim/twin/align/joint_fit/overlays.py
Sim silhouettes exclude the URDF jaws (real fingers are TPU fin-ray). Wrist: the top camera's blue (table plane) cast
into the wrist view (red joe_cal2, green C; real blue outlined light blue).
Writes sim/twin/out/align/joint_fit/overlay_{top,inspect,table}.jpg (8 held-out frames each, 2x4 tiles).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model as M  # noqa: E402
import table as TB  # noqa: E402
from stage2 import th_from  # noqa: E402  (real mask tinted blue, don't-care fingers tinted yellow)

OUT = M.TWIN / "out/align/joint_fit"
RED, GREEN = (255, 50, 50), (40, 255, 40)


def up(m):
    return cv2.resize(m, (640, 480), interpolation=cv2.INTER_NEAREST)


def outline(img, m, col, t=2):
    cs, _ = cv2.findContours(up(m), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cs, -1, col, t)


def main():
    items, D = M.load_data()
    rob = M.Robot()
    r = json.loads((OUT / "fit.json").read_text())
    old, new = M.zero_theta(), th_from(r["theta"]["C"])
    rgb = D["rgb"]
    for cam in ("top", "inspect"):
        sel = [it for it in items if it["cam"] == cam and it["held"]]
        if cam == "wrist":
            sel = [it for it in sel if it["real"][:110].mean() <= 0.01]
        scored = []
        for it in sel:
            lo = M.frame_loss(rob, cam, it["state"], it["real"], it["dt"], old, want_masks=True, dc=it["dc"])
            ln = M.frame_loss(rob, cam, it["state"], it["real"], it["dt"], new, want_masks=True, dc=it["dc"])
            if lo[2] and ln[2] and lo[3].sum() > (300 if cam != "wrist" else 100):
                scored.append((it, lo, ln))
        pick = scored[:: max(1, len(scored) // 8)][:8]
        tiles = []
        for it, lo, ln in pick:
            img = cv2.resize(rgb[it["i"]], (640, 480), interpolation=cv2.INTER_LINEAR)
            rm = up(it["real"]) > 0
            img[rm] = (0.55 * img[rm] + [0, 0, 100]).astype(np.uint8)
            dcm = (up(it["dc"]) > 0) & ~rm
            img[dcm] = (0.7 * img[dcm] + [60, 60, 0]).astype(np.uint8)
            outline(img, lo[3], RED)
            outline(img, ln[3], GREEN)
            cv2.putText(img, f"ep {it['ep']} fr {it['fr']}  IoU {lo[1]:.2f} -> {ln[1]:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            tiles.append(cv2.resize(img, (480, 360), interpolation=cv2.INTER_AREA))
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        g = np.concatenate([np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles), 4)], 0)
        cv2.imwrite(str(OUT / f"overlay_{cam}.jpg"), cv2.cvtColor(g, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(cam, len(pick), "tiles", flush=True)
    # table view: top-camera blue cast into the wrist view
    tis = [t for t in TB.build(items, D) if t["held"]]
    h_new = 0.01 + new["plane"][0]
    tiles = []
    for t in tis[:: max(1, len(tis) // 8)][:8]:
        img = cv2.resize(rgb[t["i"]], (640, 480), interpolation=cv2.INTER_LINEAR)
        outline(img, t["wrist_blue"], (80, 160, 255), 1)
        outline(img, TB.overlay(rob, t, old, None), RED)
        outline(img, TB.overlay(rob, t, new, None, h_new), GREEN)
        lo, ln = TB.loss(rob, t, old)[0], TB.loss(rob, t, new, h_new)[0]
        cv2.putText(img, f"ep {t['ep']} fr {t['fr']}  loss {lo:.1f} -> {ln:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)
        tiles.append(cv2.resize(img, (480, 360), interpolation=cv2.INTER_AREA))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    g = np.concatenate([np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles), 4)], 0)
    cv2.imwrite(str(OUT / "overlay_table.jpg"), cv2.cvtColor(g, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("OVERLAYS_OK", flush=True)


if __name__ == "__main__":
    main()
