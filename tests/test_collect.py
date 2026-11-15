"""collect.Store + build_dataset (split by spin), and Spin & save through the loop controller."""

import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "smolvla"))
import collect as C  # noqa: E402


def fake_spin(root: Path, name: str, n: int = 12) -> Path:
    d = root / name
    d.mkdir(parents=True)
    for i in range(n):
        (d / f"crop_{i:02d}.jpg").write_bytes(b"jpg" + bytes([i]))
    (d / "sweep.json").write_text(json.dumps({"inspection": {"per_frame": [
        {"i": i, "label": "perfect" if i % 2 else "defective", "mark": None if i % 2 else "circle", "score": 0.9}
        for i in range(n)]}}))
    return d


def test_store_round_trip_and_exclusions(tmp_path):
    st = C.Store(tmp_path / "ds")
    e = st.add("circle", [str(fake_spin(tmp_path, "s1"))], pred="circle")
    st.update(e["id"], toggle=3)
    st.update(e["id"], toggle=5)
    st.update(e["id"], toggle=3)
    again = C.Store(tmp_path / "ds")
    assert again.data["entries"][0]["excluded"] == [5]
    assert again.counts()["circle"] == {"spins": 1, "crops": 11}
    v = again.view()[0]
    assert len(v["frames"]) == 12 and v["frames"][0]["pred"] == "circle" and v["frames"][1]["pred"] == "plain"
    again.update(e["id"], label="cross")
    assert again.counts()["cross"]["spins"] == 1
    with pytest.raises(ValueError):
        again.add("purple", [])
    again.update(e["id"], delete=True)
    assert again.counts()["cross"]["spins"] == 0


def test_build_dataset_splits_by_spin(tmp_path):
    st = C.Store(tmp_path / "ds")
    for lab in C.LABELS:
        for k in range(10):
            e = st.add(lab, [str(fake_spin(tmp_path, f"{lab}{k}"))])
            if k == 0:
                st.update(e["id"], toggle=0)
    m = C.build_dataset(tmp_path / "out", st)
    spins = {}
    for r in m["images"]:
        spins.setdefault(r["spin"], set()).add(r["split"])
    assert all(len(v) == 1 for v in spins.values())  # no spin crosses splits
    parts = {r["split"] for r in m["images"]}
    assert parts == {"train", "val", "test"}
    assert len(m["images"]) == 3 * (10 * 12 - 1)
    assert (tmp_path / "out" / "train" / "plain").is_dir()
    with pytest.raises(FileExistsError):
        C.build_dataset(tmp_path / "out", st)
