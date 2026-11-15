"""Real | before | after close-ups from two evaluation renders (plain Python).

    python sim/twin/visuals/fit/closeups.py --before before --after after [--samples 16 46 81 ...]

For each chosen held-out sample and camera: the image crop with the most table in it (region mask bit 1 of the
`after` render), 2x enlarged, as real | sim before | sim after (camera response applied to both sims), plus full
frames. Writes sim/twin/out/visual_fit/closeup_<cam>.jpg and compare_<cam>.jpg.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from sim.twin.visuals.fit import eval_visuals as ev  # noqa: E402

OUT = ev.OUT
CW, CH = 240, 180


def label(img, text):
    """Copy of `img` with a caption."""
    img = np.ascontiguousarray(img)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    """Compose the close-up and full-frame comparison sheets."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="before")
    ap.add_argument("--after", default="after")
    ap.add_argument("--samples", type=int, nargs="*", default=None)
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    samples, real = ev.load_eval()
    zb, za = np.load(OUT / f"render_{a.before}.npz"), np.load(OUT / f"render_{a.after}.npz")
    held = [i for i, s in enumerate(samples) if s["split"] == "held"]
    ids = a.samples or held[2:: max(1, len(held) // a.n)][: a.n]
    for cam in ev.CAMS:
        rows, full = [], []
        for i in ids:
            before = ev.apply_ccm({cam: zb[cam][i:i + 1]})[cam][0]
            after = ev.apply_ccm({cam: za[cam][i:i + 1]})[cam][0]
            table = ((za["mask_" + cam][i] & 1) > 0).astype(np.float32)
            score = cv2.boxFilter(table, -1, (CW, CH), normalize=True)[CH // 2:-CH // 2, CW // 2:-CW // 2]
            y, x = np.unravel_index(np.argmax(score), score.shape)
            tag = f"ep{samples[i]['ep']} {samples[i]['moment']}"
            crops = [cv2.resize(im[y:y + CH, x:x + CW], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                     for im in (real[cam][i], before, after)]
            rows.append(np.concatenate([label(c, t) for c, t in zip(
                crops, (f"real {cam} {tag}", "sim before", "sim after"), strict=True)], 1))
            full.append(np.concatenate([label(c, t) for c, t in zip(
                (real[cam][i], before, after), (f"real {cam} {tag}", "sim before", "sim after"), strict=True)], 1))
        cv2.imwrite(str(OUT / f"closeup_{cam}.jpg"), np.concatenate(rows, 0)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(str(OUT / f"compare_{cam}.jpg"), np.concatenate(full, 0)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("CLOSEUPS_OK", ids)


if __name__ == "__main__":
    main()
