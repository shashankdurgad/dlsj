"""dlsj.deploy.studio_registry against a fake Physical AI Studio API, and deploy/studio/build_dataset.py."""

import hashlib
import io
import json
import sys
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dlsj.deploy import studio_registry as SR

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "smolvla"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "studio"))
import build_dataset as BD  # noqa: E402

MID, PID, DID = "11111111-1111-1111-1111-111111111111", "22222222-2222-2222-2222-222222222222", \
    "33333333-3333-3333-3333-333333333333"
FILES = {"manifest.json": b'{"format": "policy_package"}', "m.xml": b"<xml/>", "m.bin": b"\x00" * 1000}


def _record(files=FILES):
    return {"model": {"id": MID, "name": "MolmoAct2 test", "policy": "molmoact2", "version": 1, "project_id": PID,
                      "dataset_id": DID, "available_backends": ["openvino"]},
            "exports": [{"type": "openvino", "size_bytes": sum(map(len, files.values())), "file_count": len(files),
                         "exported_at": "2026-11-15T00:00:00Z"}]}


@pytest.fixture
def studio():
    state = {"record": _record(), "downloads": 0}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == f"/api/models/{MID}":
                return self._json(state["record"])
            if self.path == f"/api/projects/{PID}":
                return self._json({"id": PID, "name": "dlsj brick sorting"})
            if self.path == f"/api/dataset/{DID}":
                return self._json({"id": DID, "name": "cal2 3-stage"})
            if self.path == f"/api/projects/{PID}/models":
                return self._json([state["record"]["model"]])
            if self.path == f"/api/models/{MID}/exports/openvino/download":
                state["downloads"] += 1
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w") as z:
                    for n, b in FILES.items():
                        z.writestr(f"exports/openvino/{n}", b)
                body = buf.getvalue()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                return self.wfile.write(body)
            self._json({"detail": "not found"}, 404)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    state["client"] = SR.Studio(f"http://127.0.0.1:{srv.server_address[1]}")
    yield state
    srv.shutdown()


def _export(d: Path, files=FILES) -> Path:
    d.mkdir(parents=True)
    for n, b in files.items():
        (d / n).write_bytes(b)
    return d


def _sums(d: Path, path: Path) -> Path:
    path.write_text("".join(f"{hashlib.sha256((d / n).read_bytes()).hexdigest()}  {n}\n" for n in FILES))
    return path


def test_seed_links_verifies_and_resolves(studio, tmp_path):
    src = _export(tmp_path / "rtc_int8")
    d = SR.seed(MID, src, studio=studio["client"], root=tmp_path / "mirror",
                studio_sums=_sums(src, tmp_path / "sums.txt"), lineage={"trained_with": "physicalai"})
    assert (d / "m.bin").stat().st_ino == (src / "m.bin").stat().st_ino  # hard link, no copy
    path, rec = SR.resolve(MID, studio=studio["client"], root=tmp_path / "mirror", allow_download=False)
    assert path == d and rec["online"] and rec["sha256_verified"]
    assert rec["name"] == "MolmoAct2 test" and rec["dataset_name"] == "cal2 3-stage"
    assert rec["project_name"] == "dlsj brick sorting" and rec["lineage"] == {"trained_with": "physicalai"}
    assert SR.record_for_export(path)["model_id"] == MID
    assert studio["downloads"] == 0


def test_seed_refuses_files_that_differ_from_studio(studio, tmp_path):
    src = _export(tmp_path / "rtc_int8")
    sums = _sums(src, tmp_path / "sums.txt")
    (src / "m.bin").write_bytes(b"\x01" * 1000)  # same size, different content
    with pytest.raises(SR.StudioError, match="differs from Studio"):
        SR.seed(MID, src, studio=studio["client"], root=tmp_path / "mirror", studio_sums=sums)


def test_resolve_downloads_missing_or_stale_copy(studio, tmp_path):
    root = tmp_path / "mirror"
    path, rec = SR.resolve(MID, studio=studio["client"], root=root)
    assert studio["downloads"] == 1 and sorted(p.name for p in path.iterdir()) == sorted(FILES)
    (path / "m.bin").write_bytes(b"short")  # truncated file -> re-download
    SR.resolve(MID, studio=studio["client"], root=root)
    assert studio["downloads"] == 2 and (path / "m.bin").stat().st_size == 1000


def test_resolve_offline_uses_saved_record_and_still_checks_files(studio, tmp_path):
    root = tmp_path / "mirror"
    path, _ = SR.resolve(MID, studio=studio["client"], root=root)
    offline = SR.Studio("http://127.0.0.1:9", timeout=0.5)
    p2, rec = SR.resolve(MID, studio=offline, root=root)
    assert p2 == path and rec["online"] is False and rec["name"] == "MolmoAct2 test"
    (path / "m.xml").unlink()
    with pytest.raises(SR.StudioError):
        SR.resolve(MID, studio=offline, root=root)


def test_resolve_without_studio_or_copy_fails(tmp_path):
    with pytest.raises(SR.StudioError, match="unreachable"):
        SR.resolve(MID, studio=SR.Studio("http://127.0.0.1:9", timeout=0.5), root=tmp_path)


def test_loop_dashboard_reads_the_studio_record(studio, tmp_path):
    import loop_control as LC

    path, _ = SR.resolve(MID, studio=studio["client"], root=tmp_path / "mirror")
    info = LC.studio_info(["--export", str(path), "--device", "GPU"])
    assert info["name"] == "MolmoAct2 test" and info["device"] == "GPU" and info["policy"] == "molmoact2"
    assert LC.studio_info(["--export", str(tmp_path)]) == {}  # a plain export folder: no Studio line
    assert LC.studio_info([]) == {}


# ---- build_dataset -------------------------------------------------------------------------------------------

CAMS = ("observation.images.wrist", "observation.images.top")


def _dataset(root: Path, top_width: int, tasks: list[str], task_index: list[int]):
    """Two episodes (3 + 2 frames) in one data file and one video file per camera."""
    feats = {"action": {"dtype": "float32", "shape": [2]}, "task_index": {"dtype": "int64", "shape": [1]}}
    for c in CAMS:
        w = top_width if c.endswith("top") else 640
        feats[c] = {"dtype": "video", "shape": [480, w, 3], "info": {"video.width": w}}
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    info = {"total_episodes": 2, "total_frames": 5, "fps": 30, "features": feats}
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/stats.json").write_text(json.dumps({c: {"mean": [top_width if c.endswith("top") else 1]}
                                                      for c in CAMS}))
    pd.DataFrame({"task_index": range(len(tasks))}, index=pd.Index(tasks, name="task")).to_parquet(
        root / "meta/tasks.parquet")
    ep = {"episode_index": [0, 1], "length": [3, 2], "dataset_from_index": [0, 3], "dataset_to_index": [3, 5]}
    for c in CAMS:
        ep |= {f"videos/{c}/chunk_index": [0, 0], f"videos/{c}/file_index": [0, 0],
               f"videos/{c}/from_timestamp": [0.0, 0.1], f"videos/{c}/to_timestamp": [0.1, 0.167],
               f"stats/{c}/mean": [[float(top_width)], [float(top_width)]]}
    pd.DataFrame(ep).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet")
    pd.DataFrame({"action": [np.array([i, i], np.float32) for i in range(5)], "task_index": task_index,
                  "episode_index": [0, 0, 0, 1, 1]}).to_parquet(root / "data/chunk-000/file-000.parquet")
    return root


def test_build_dataset_relabels_source_videos(tmp_path):
    src = _dataset(tmp_path / "src", 640, ["pick up the brick"], [0] * 5)
    for c in CAMS:
        (src / f"videos/{c}/chunk-000").mkdir(parents=True)
        (src / f"videos/{c}/chunk-000/file-000.mp4").write_bytes(b"video " + c.encode())
    (src / "observation.images.inspect").mkdir()
    labels = _dataset(tmp_path / "labels", 560, ["pick up the blue brick", "bring the brick to the inspection camera"],
                      [0, 1, 1, 0, 1])
    (labels / "MANIFEST.json").write_text("{}")
    out = BD.build(labels, src, tmp_path / "out", ["observation.images.top"])
    o = Path(out["out"])
    assert out["tasks"] == ["bring the brick to the inspection camera", "pick up the blue brick"]
    assert pd.read_parquet(o / "data/chunk-000/file-000.parquet")["task_index"].tolist() == [0, 1, 1, 0, 1]
    info = json.loads((o / "meta/info.json").read_text())
    assert info["features"]["observation.images.top"]["shape"] == [480, 640, 3]  # full frame, from the source
    assert json.loads((o / "meta/stats.json").read_text())["observation.images.top"]["mean"] == [640]
    ep = pd.read_parquet(o / "meta/episodes/chunk-000/file-000.parquet")
    assert [list(v) for v in ep["stats/observation.images.top/mean"]] == [[640.0], [640.0]]
    v = o / "videos/observation.images.wrist/chunk-000/file-000.mp4"
    assert v.stat().st_ino == (src / "videos/observation.images.wrist/chunk-000/file-000.mp4").stat().st_ino
    assert (o / "meta/training/MANIFEST.json").exists() and not (src / "meta/training").exists()


def test_build_dataset_refuses_a_different_recording(tmp_path):
    src = _dataset(tmp_path / "src", 640, ["t"], [0] * 5)
    labels = _dataset(tmp_path / "labels", 560, ["t"], [0] * 5)
    df = pd.read_parquet(labels / "data/chunk-000/file-000.parquet")
    df["action"] = [np.array([9, 9], np.float32) if i == 2 else a for i, a in enumerate(df["action"])]
    df.to_parquet(labels / "data/chunk-000/file-000.parquet")
    with pytest.raises(ValueError, match="column action differs"):
        BD.build(labels, src, tmp_path / "out", ["observation.images.top"])


def test_build_dataset_as_recorded_drops_depth(tmp_path):
    src = _dataset(tmp_path / "src", 640, ["pick up the brick"], [0] * 5)
    for c in CAMS:
        (src / f"videos/{c}/chunk-000").mkdir(parents=True)
        (src / f"videos/{c}/chunk-000/file-000.mp4").write_bytes(b"v")
    out = BD.build(src, src, tmp_path / "out", [], drop=["observation.images.top"])
    o = Path(out["out"])
    assert out["cameras"] == ["observation.images.wrist"] and not (o / "videos/observation.images.top").exists()
    assert "observation.images.top" not in json.loads((o / "meta/info.json").read_text())["features"]
    assert "observation.images.top" not in json.loads((o / "meta/stats.json").read_text())
    ep = pd.read_parquet(o / "meta/episodes/chunk-000/file-000.parquet")
    assert not [c for c in ep.columns if "images.top" in c] and "videos/observation.images.wrist/file_index" in ep
    assert "observation.images.top" in json.loads((src / "meta/info.json").read_text())["features"]  # source intact
