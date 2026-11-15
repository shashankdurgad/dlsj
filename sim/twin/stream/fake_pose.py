"""Fake follower pose server for testing the live mirror (stdlib only, runs with any python3):

    python3 sim/twin/stream/fake_pose.py [--port 8041] [--host 127.0.0.1]
    curl http://127.0.0.1:8041/api/pose   ->  {"t": ..., "q": [pan, lift, elbow, wrist_flex, wrist_roll, gripper],
                                                "frame": "cal2", "age": 0.0}

Smooth sinusoidal motion (LeRobot degrees, cal2 frame, gripper 0-100) inside the range the recorded cal2 episodes
cover, with incommensurate periods so the arm never repeats exactly. Same schema as the robot laptop's /api/pose.
"""
import argparse
import json
import math
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# (centre, amplitude, period s) per joint: pan, lift, elbow, wrist_flex, wrist_roll, gripper
MOTION = [(15.0, 35.0, 11.0), (-45.0, 30.0, 7.3), (25.0, 30.0, 9.1), (70.0, 20.0, 5.7), (15.0, 45.0, 13.0),
          (35.0, 25.0, 4.3)]


def pose(t: float) -> list:
    return [round(c + a * math.sin(2 * math.pi * t / p + k), 3) for k, (c, a, p) in enumerate(MOTION)]


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive, like the robot laptop's server

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.split("?")[0] != "/api/pose":
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        t = time.time()
        body = json.dumps({"t": round(t, 3), "q": pose(t), "frame": "cal2", "age": 0.0}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8041)
    a = ap.parse_args()
    print(f"fake pose server on http://{a.host}:{a.port}/api/pose", flush=True)
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()
