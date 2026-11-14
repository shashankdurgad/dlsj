"""Inside the Studio container: the backend code path "Run model" uses, on an imported model (no robot, no cameras).

Calls the backend's own `runtime_session.model.model_inputs` (the check Studio does before it loads a model into an
environment) and `workers.model_worker.Inference.infer` (what the model worker serves on `rpc/model/infer`), with the
camera store replaced by recorded full 640x480 frames, and compares the chunks with the PyTorch reference.

    docker cp <dir with this file + act_warp_417_ref.npz> physical-ai-studio-xpu:/tmp/act_ctr
    docker exec physical-ai-studio-xpu nice -n 5 python /tmp/act_ctr/backend_check.py \
        /app/storage/models/<model id>/exports/openvino /tmp/act_ctr/act_warp_417_ref.npz GPU
"""
import json, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, "/app/application/backend/src")
from runtime_session.model import model_inputs
from workers.model_worker import Inference, InferRequest
from physicalai.inference import InferenceModel

export = Path(sys.argv[1]); ref = dict(np.load(sys.argv[2])); device = sys.argv[3]
spec = SimpleNamespace(joint_names=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"],
                       observations={"wrist": None, "top": None, "inspect": None})  # the so101 brick station cameras
inputs = model_inputs(export, spec)
print("model_inputs:", inputs)

class FakeStore:  # stands in for the shared-memory camera store: latest full 640x480 RGB frame per camera
    def __init__(self, i): self.i = i
    def read_observation(self, cam): return SimpleNamespace(value=ref[cam][self.i])

model = InferenceModel(export, backend="openvino", device=device)
outs, ts = [], []
for i in range(len(ref["state"])):
    inf = Inference(model, FakeStore(i), inputs.images)
    model.reset(); t = time.perf_counter()
    chunk = inf.infer(InferRequest(state=ref["state"][i].tolist()))
    ts.append((time.perf_counter() - t) * 1e3)
    outs.append(np.asarray(chunk.actions, np.float32))
pred = np.stack(outs); d = np.abs(pred - ref["ref"])
print(json.dumps({"device": device, "chunk_shape": list(outs[0].shape), "frames": len(outs), "max_abs": round(float(d.max()), 4),
                  "mean_abs": round(float(d.mean()), 5), "median_ms_incl_rpc_payload": round(float(np.median(ts[3:])), 1)}))
