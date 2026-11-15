import json
import math
from pathlib import Path
from urllib.request import urlopen

from dlsj.demo_board import Board, _plan, link_frames, load_context, render_page


def test_board_uses_held_out_measurements():
    ctx = load_context()
    page = render_page(ctx)
    assert ctx["balanced"] == "92.9"
    assert ctx["episodes"] == 814
    assert ctx["pick_only"] == 133
    assert "92.9" in page
    assert "1,035" in page
    assert "not measured yet" in page
    assert "14 Nov" in page
    assert "133 stop at the camera" in page
    assert "9 of 16" in page
    assert "283" not in page
    assert "212 ms" not in page
    assert ctx["cal2"] == 431 and ctx["kept"] == 250
    assert ctx["yellow"] == 229 and ctx["orange"] == 201 and ctx["purple"] == 251
    assert _plan([[1, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 0]]) == [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    assert _plan([[1, 2, 3]]) is None
    assert "467" not in page
    assert "140 ms" not in page


def test_board_serves_the_page_and_an_idle_loop():
    board = Board("127.0.0.1", 0)
    thread_target = board.server.serve_forever
    import threading
    threading.Thread(target=thread_target, daemon=True).start()
    try:
        with urlopen(f"http://127.0.0.1:{board.port}/") as resp:
            body = resp.read().decode()
        assert resp.status == 200
        assert "Detect" in body
        assert 'id="arm"' in body
        assert "All 814 runs, together" in body
        assert "demo_arm.js" in body
        with urlopen(f"http://127.0.0.1:{board.port}/demo_arm.js") as arm_js:
            script = arm_js.read().decode()
        assert arm_js.status == 200
        assert "echo" in script
        with urlopen(f"http://127.0.0.1:{board.port}/so101/kin.json") as kin:
            payload = json.load(kin)
        assert payload["visuals"]
        assert payload["flock"] >= 32
        with urlopen(f"http://127.0.0.1:{board.port}/vendor/three.module.js") as three:
            assert three.status == 200
            assert len(three.read()) > 100_000
        with urlopen(f"http://127.0.0.1:{board.port}/api/arm") as arm:
            assert "traces" in json.load(arm)
        with urlopen(f"http://127.0.0.1:{board.port}/api/live") as resp:
            snap = json.load(resp)
        assert snap["live"] is False
        assert snap["policy"] == "π0.5"
        with urlopen(f"http://127.0.0.1:{board.port}/live") as resp:
            live = resp.read().decode()
        assert resp.status == 200
        assert "STOP" in live and "/live/state" in live and "/ctl/state" not in live
        # Operator tools from the 8020 page live in a drawer; their commands are proxied.
        assert 'class="drawer"' in live
        assert "Inspection pose" in live and "Data collection" in live
        # The target brick stays internal to the loop; the page never draws it.
        assert "pc.target" not in live and "TARGET" not in live
        assert "467" not in live
    finally:
        board.server.shutdown()


def test_so101_zero_pose_reaches_out():
    kin = json.loads((Path(__file__).parents[1] / "dlsj/so101_assets/kin.json").read_text())
    tip = link_frames(kin["joints"], [0, 0, 0, 0, 0, 0])["gripper_link"]
    dist = math.sqrt(tip[12] ** 2 + tip[13] ** 2 + tip[14] ** 2)
    assert 0.2 < dist < 0.55
    assert tip[14] > 0.05


def test_loop_console_page_uses_same_origin_routes():
    from dlsj.live_console import page

    html = page(proxy=False)
    assert "/ctl/state" in html and "/live/state" not in html
    assert "__ROUTES__" not in html and "pc.target" not in html and "TARGET" not in html
