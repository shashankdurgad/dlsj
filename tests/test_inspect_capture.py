import csv
import json
import re
import shutil
import subprocess
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import cv2
import numpy as np
import pytest

from dlsj import inspect_capture
from dlsj.inspect_capture import PAGE, Capture, Syncer, make_handler


class FakeStream:
    def __init__(self, h=48, w=64):
        self.name = "inspect"
        self.frame = None
        self.seq = 0
        self.h, self.w = h, w

    def new_frame(self, value=120):
        self.frame = np.full((self.h, self.w, 3), value, dtype=np.uint8)
        self.seq += 1

    def info(self):
        return {"name": self.name, "index": 0, "fps": 30.0, "size": f"{self.w}x{self.h}", "error": None}


@pytest.fixture
def cap(tmp_path):
    return Capture(FakeStream(), tmp_path / "out", burst_fps=5, session="20260101_120000")


def rows(c):
    with (c.dir / "index.csv").open() as f:
        return list(csv.DictReader(f))


def test_save_writes_jpeg_and_index_row(cap):
    assert cap.save() is None  # no frame yet
    cap.stream.new_frame()
    path = cap.save()
    assert path == cap.dir / "in_view" / "00000.jpg"
    img = cv2.imread(str(path))
    assert img.shape == (48, 64, 3)
    (row,) = rows(cap)
    assert row["file"] == "in_view/00000.jpg" and row["tag"] == "in_view"
    assert (row["width"], row["height"]) == ("64", "48") and float(row["time"]) > 0


def test_tags_route_to_folders(cap):
    cap.stream.new_frame()
    cap.command("save")
    cap.command("not_in_view")
    for _ in range(2):
        cap.stream.new_frame()
        cap.command("save")
    cap.command("in_view")
    cap.stream.new_frame()
    cap.command("save")
    assert sorted(p.name for p in (cap.dir / "in_view").iterdir()) == ["00000.jpg", "00001.jpg"]
    assert sorted(p.name for p in (cap.dir / "not_in_view").iterdir()) == ["00000.jpg", "00001.jpg"]
    assert [r["tag"] for r in rows(cap)] == ["in_view", "not_in_view", "not_in_view", "in_view"]
    assert cap.counts == {"in_view": 2, "not_in_view": 2}


def test_same_frame_never_saved_twice(cap):
    cap.stream.new_frame()
    assert cap.save() is not None
    assert cap.save() is None
    cap.command("not_in_view")
    assert cap.save() is None  # a tag change does not make the frame new
    assert len(rows(cap)) == 1


def test_burst_toggles_and_saves_new_frames(cap):
    assert not cap.bursting
    cap.command("burst")
    assert cap.bursting
    cap.burst_fps = 200
    t = threading.Thread(target=cap.run, daemon=True)
    t.start()
    try:
        for v in range(3):
            cap.stream.new_frame(v)
            for _ in range(200):
                if cap.counts["in_view"] == v + 1:
                    break
                threading.Event().wait(0.01)
        assert cap.counts["in_view"] == 3
    finally:
        cap.command("burst")
        assert not cap.bursting
        cap.stop()
        t.join(2)
    with pytest.raises(ValueError):
        cap.command("nope")


def test_http_endpoints(cap):
    cap.stream.new_frame()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler({"inspect": cap.stream}, cap))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    post = lambda c: urllib.request.urlopen(urllib.request.Request(base + "/cap/" + c, method="POST"))
    try:
        assert "<title>Inspect capture</title>" in urllib.request.urlopen(base + "/").read().decode()
        post("not_in_view")
        post("save")
        post("burst")
        state = json.loads(urllib.request.urlopen(base + "/cap/state").read())
        assert state["tag"] == "not_in_view" and state["bursting"]
        assert state["counts"] == {"in_view": 0, "not_in_view": 1}
        assert state["camera"]["size"] == "64x48" and state["sync"] is None
        assert state["folder"].endswith("20260101_120000")
        with pytest.raises(urllib.error.HTTPError):
            post("delete_everything")
        assert json.loads(urllib.request.urlopen(base + "/state").read())["cameras"][0]["name"] == "inspect"
    finally:
        srv.shutdown()


class FakeRun:
    def __init__(self, code=0):
        self.calls, self.code = [], code

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        return subprocess.CompletedProcess(cmd, self.code, "", "permission denied" if self.code else "")


def test_sync_commands(tmp_path, monkeypatch):
    run = FakeRun()
    monkeypatch.setattr(inspect_capture.subprocess, "run", run)
    s = Syncer(tmp_path / "out", "gb10:desmond/inspect_frames/", ssh_config="/x/ssh_config")
    assert s.sync_once() and s.sync_once()
    assert run.calls[0] == ["ssh", "-F", "/x/ssh_config", "gb10", "mkdir", "-p", "desmond/inspect_frames"]
    rsyncs = [c for c in run.calls if c[0] == "rsync"]
    assert len(rsyncs) == 2 and sum(c[0] == "ssh" for c in run.calls) == 1  # mkdir only before the first rsync
    assert rsyncs[0] == ["rsync", "-az", "-e", "ssh -F /x/ssh_config", f"{tmp_path / 'out'}/", "gb10:desmond/inspect_frames/"]
    assert not any("--delete" in a for c in run.calls for a in c)
    assert s.snapshot()["last_ok"] and s.snapshot()["error"] is None


def test_sync_failure_is_reported(tmp_path, monkeypatch):
    run = FakeRun(code=255)
    monkeypatch.setattr(inspect_capture.subprocess, "run", run)
    s = Syncer(tmp_path / "out", "gb10:desmond/x/")
    assert not s.sync_once()
    assert "permission denied" in s.snapshot()["error"] and s.snapshot()["last_ok"] is None
    assert not any("--delete" in a for c in run.calls for a in c)


@pytest.mark.parametrize("dest", ["gb10:inspect_frames/", "gb10:/tmp/x", "gb10:desmond/../x", "nohost", ":desmond/x"])
def test_sync_only_under_desmond(tmp_path, dest):
    with pytest.raises(ValueError):
        Syncer(tmp_path, dest)


def test_page_javascript_parses(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    js = tmp_path / "page.js"
    js.write_text(re.search(r"<script>(.*)</script>", PAGE, re.S).group(1))
    r = subprocess.run([node, "--check", str(js)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
