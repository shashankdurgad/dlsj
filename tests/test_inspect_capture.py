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
    return Capture(FakeStream(), tmp_path / "out", session="20260101_120000")


def rows(c):
    with (c.dir / "index.csv").open() as f:
        return list(csv.DictReader(f))


def test_key_captures_jpeg_and_index_row(cap):
    assert cap.command("in_view") is None  # no frame yet
    cap.stream.new_frame()
    path = cap.command("in_view")
    assert path == cap.dir / "in_view" / "00000.jpg"
    img = cv2.imread(str(path))
    assert img.shape == (48, 64, 3)
    (row,) = rows(cap)
    assert row["file"] == "in_view/00000.jpg" and row["tag"] == "in_view"
    assert (row["width"], row["height"]) == ("64", "48") and float(row["time"]) > 0
    assert cap.last_tag == "in_view"


def test_each_key_captures_into_its_own_folder(cap):
    for tag in ["in_view", "not_in_view", "not_in_view", "in_view"]:
        cap.stream.new_frame()
        cap.command(tag)
    assert sorted(p.name for p in (cap.dir / "in_view").iterdir()) == ["00000.jpg", "00001.jpg"]
    assert sorted(p.name for p in (cap.dir / "not_in_view").iterdir()) == ["00000.jpg", "00001.jpg"]
    assert [r["tag"] for r in rows(cap)] == ["in_view", "not_in_view", "not_in_view", "in_view"]
    assert cap.counts == {"in_view": 2, "not_in_view": 2}


def test_holding_a_key_never_saves_the_same_frame_twice(cap):
    cap.stream.new_frame()
    assert cap.command("in_view") is not None
    assert cap.command("in_view") is None  # key repeat faster than the camera
    assert cap.command("not_in_view") is None  # the other key does not make the frame new either
    assert len(rows(cap)) == 1
    with pytest.raises(ValueError):
        cap.command("burst")


def test_http_endpoints(cap):
    cap.stream.new_frame()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler({"inspect": cap.stream}, cap))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    post = lambda c: urllib.request.urlopen(urllib.request.Request(base + "/cap/" + c, method="POST"))
    try:
        assert "<title>Inspect capture</title>" in urllib.request.urlopen(base + "/").read().decode()
        post("not_in_view")
        state = json.loads(urllib.request.urlopen(base + "/cap/state").read())
        assert state["last_tag"] == "not_in_view"
        assert state["counts"] == {"in_view": 0, "not_in_view": 1}
        assert state["camera"]["size"] == "64x48" and state["sync"] is None
        assert state["folder"].endswith("20260101_120000")
        for gone in ("burst", "save", "delete_everything"):
            with pytest.raises(urllib.error.HTTPError):
                post(gone)
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


def test_undo_deletes_latest_capture_and_walks_back(cap):
    for tag in ["in_view", "not_in_view", "in_view"]:
        cap.stream.new_frame()
        cap.command(tag)
    third = cap.dir / "in_view" / "00001.jpg"
    assert third.exists()
    assert cap.command("undo") == third and not third.exists()
    assert cap.counts == {"in_view": 1, "not_in_view": 1} and cap.last_tag == "not_in_view"
    assert [r["file"] for r in rows(cap)] == ["in_view/00000.jpg", "not_in_view/00000.jpg"]
    assert cap.command("undo") == cap.dir / "not_in_view" / "00000.jpg"
    assert cap.counts == {"in_view": 1, "not_in_view": 0}
    cap.stream.new_frame()
    assert cap.command("in_view") == cap.dir / "in_view" / "00002.jpg"  # numbers are never reused
    assert cap.command("undo") and cap.command("undo") and cap.command("undo") is None  # nothing left
    assert cap.counts == {"in_view": 0, "not_in_view": 0} and rows(cap) == []


def test_undo_removes_the_remote_copy_before_the_next_rsync(cap, monkeypatch):
    run = FakeRun()
    monkeypatch.setattr(inspect_capture.subprocess, "run", run)
    cap.syncer = Syncer(cap.out, "gb10:desmond/inspect_frames/", ssh_config="/x/ssh_config")
    cap.stream.new_frame()
    cap.command("in_view")
    assert cap.syncer.sync_once()
    cap.command("undo")
    assert cap.syncer.sync_once()
    rm = [c for c in run.calls if "rm" in c]
    assert rm == [["ssh", "-F", "/x/ssh_config", "gb10", "rm", "-f", "--",
                   "desmond/inspect_frames/20260101_120000/in_view/00000.jpg"]]
    assert run.calls.index(rm[0]) < len(run.calls) - 1 and run.calls[-1][0] == "rsync"  # delete, then rsync
    assert cap.syncer.sync_once() and len([c for c in run.calls if "rm" in c]) == 1  # sent once
    assert not any("--delete" in a for c in run.calls for a in c)
    with pytest.raises(ValueError):
        cap.syncer.queue_delete("../../etc/passwd")
    with pytest.raises(ValueError):
        cap.syncer.queue_delete("/abs/path.jpg")


def test_undo_over_http(cap):
    cap.stream.new_frame()
    cap.command("in_view")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler({"inspect": cap.stream}, cap))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert json.loads(urllib.request.urlopen(base + "/cap/state").read())["can_undo"]
        urllib.request.urlopen(urllib.request.Request(base + "/cap/undo", method="POST"))
        state = json.loads(urllib.request.urlopen(base + "/cap/state").read())
        assert state["counts"]["in_view"] == 0 and not state["can_undo"]
    finally:
        srv.shutdown()
