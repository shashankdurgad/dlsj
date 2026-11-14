"""Top-camera <-> table-plane mapping from Joe's camera-fit.json (pinhole, world_to_cv, position), plus checks.

project(X): x_cv = R (X - C); u = f x/z + cx, v = f y/z + cy.   backproject(u, v): ray from C hitting z = 0.
"""

import json

import numpy as np


class TopCamera:
    def __init__(self, fit: dict):
        c = fit["camera"]
        self.R = np.asarray(c["world_to_cv"], float)
        self.C = np.asarray(c["position"], float)
        self.f = float(c["focal_px"])
        self.cx, self.cy = c["principal_point"]

    def project(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        x = (X - self.C) @ self.R.T
        return np.c_[self.f * x[:, 0] / x[:, 2] + self.cx, self.f * x[:, 1] / x[:, 2] + self.cy]

    def backproject(self, uv, z=0.0):
        uv = np.atleast_2d(np.asarray(uv, float))
        d = np.c_[(uv[:, 0] - self.cx) / self.f, (uv[:, 1] - self.cy) / self.f, np.ones(len(uv))] @ self.R  # R^T d
        t = (z - self.C[2]) / d[:, 2]
        return self.C + t[:, None] * d

    def metres_per_px(self, uv):
        a = self.backproject(np.asarray(uv) + [[0, 0]])
        b = self.backproject(np.asarray(uv) + [[1, 0]])
        return float(np.linalg.norm(a - b))


if __name__ == "__main__":
    import sys

    cam = TopCamera(json.load(open(sys.argv[1])))
    print("base origin ->", cam.project([0, 0, 0]).round(1))
    print("image centre on table ->", cam.backproject([320, 240]).round(3))
