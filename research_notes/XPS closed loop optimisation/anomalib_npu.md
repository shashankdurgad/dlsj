# Anomalib v2.7.0 brick-cross defect detector on the Panther Lake NPU (OpenVINO 2026.x)

Research date: 2026-11-14. Sources: the Anomalib source at git tag `lib/v2.7.0`, which I cloned and read directly (`open-edge-platform/anomalib`; release tags are now prefixed `lib/`); the OpenVINO NPU docs taken from the `openvinotoolkit/openvino` master docs source in November 2026; and arXiv papers. File and line references below point into the v2.7.0 tree.

Staleness flags:
- Anomalib docs at readthedocs `en/v1`/`0.3.x` describe the old `optimize.py`/`export_mode` APIs. They are stale.
- Before 2.5.1 the ONNX exporter could use `dynamo=False`. 2.7.0 removed that option.
- Intel's Dobot article dates from 2023 and used Python 3.8.

## 1. Model choice: accuracy on small, colour-varied, hand-drawn crosses; data needs; NPU exportability

### Takeaway
I recommend two models for 50-200 plain-brick images:
- **PatchCore** with a small backbone such as `resnet18`, or `wide_resnet50_2` if there is latency headroom. It is the most robust to lighting and viewpoint, needs no training, and has a modest memory bank at this data size.
- **PaDiM (resnet18)** as the fastest, simplest option to put on the NPU.

**EfficientAD** has the best published accuracy and speed. It needs the ImageNette auxiliary set plus a long training run, and the RAD study did not evaluate it. **SuperSimpleNet** is worth trying second because it can also use a few labelled crosses. Dinomaly, AnomalyDINO, RAD, FoundAD and WinCLIP are ViT/DINO based (DINOv2/v3, CLIP). They are heavier and riskier on NPU compile. Keep AnomalyDINO as a few-shot baseline only.

### Cited Findings
**Models in Anomalib 2.7.0**
- The model zoo in `src/anomalib/models/image/` includes: anomaly_dino, anomalyvfm, cfa, cflow, cfm, csflow, dfkde, dfm, dinomaly, draem, dsr, efficient_ad, fastflow, foundad, fre, ganomaly, general_ad, glass, inp_former, mh_patchcore, padim, patchcore, patchflow, rad, reverse_distillation, stfpm, super_add, supersimplenet, uflow, uninet, vlm_ad and winclip — [anomalib lib/v2.7.0 tree](https://github.com/open-edge-platform/anomalib/tree/lib/v2.7.0/src/anomalib/models/image)
- v2.7.0 was released on 06 Oct (2026). It added MH-PatchCore, RAD (retrieval-based, training-free) and FoundAD. It removed the legacy ONNX exporter (`dynamo=False`), so export now needs `onnxscript` through `anomalib[openvino]`. It requires `torch>=2.10`/`torchvision>=0.25`. Checkpoint loading now defaults to `weights_only=True` — [Anomalib releases](https://github.com/open-edge-platform/anomalib/releases)
- Other releases:
  - 2.5.1 moved DINOv2 models to timm with DINOv3 support and added `dynamic_shapes` for dynamo ONNX/OpenVINO export.
  - 2.6.1 changed FastFlow to native padding.
  - [Anomalib releases](https://github.com/open-edge-platform/anomalib/releases)

**Model defaults and inference ops (v2.7.0 source)**
- **PatchCore** defaults are `backbone="wide_resnet50_2"`, `layers=("layer2","layer3")`, `coreset_sampling_ratio=0.1` and `num_neighbors=9`.
  - Inference distance is computed as `torch.matmul(x, y.T)` followed by `clamp_min_(0).sqrt_()`. The source deliberately avoids `torch.cdist` for export.
  - The nearest neighbour is found with `distances.min(1)` or `topk(k, largest=False)`.
  - The anomaly map is built with `F.interpolate` to full input size, then `GaussianBlur2d(sigma=4)`, which has kernel size 2*int(4*4+0.5)+1 = 33.
  - Sources: `patchcore/lightning_model.py:143-147`, `patchcore/torch_model.py:287-373`, `patchcore/anomaly_map.py:53-82` — [source](https://github.com/open-edge-platform/anomalib/tree/lib/v2.7.0/src/anomalib/models/image/patchcore)
- **PaDiM** defaults to `backbone="resnet18"`. Mahalanobis distance is computed as `delta=(embedding-mean)`, then `(torch.matmul(delta, inv_covariance)*delta).sum(2)`. The precomputed per-patch `inv_covariance` has shape (num_patches, C, C). The map is then upsampled and Gaussian-blurred (sigma=4). Only plain matmul, elementwise and conv ops are used — [padim/anomaly_map.py:70-98](https://github.com/open-edge-platform/anomalib/blob/lib/v2.7.0/src/anomalib/models/image/padim/anomaly_map.py)
- `GaussianBlur2d` is implemented as `F.pad` followed by a depthwise `F.conv2d` with a precomputed kernel. It exports as ordinary Pad and Convolution ops — [components/filters/blur.py:48-134](https://github.com/open-edge-platform/anomalib/blob/lib/v2.7.0/src/anomalib/models/components/filters/blur.py)
- **SuperSimpleNet** defaults to `wide_resnet50_2.tv_in1k` with layers 2-3 and uses `GaussianBlur2d(sigma=4)` in its anomaly map — [supersimplenet/torch_model.py](https://github.com/open-edge-platform/anomalib/blob/lib/v2.7.0/src/anomalib/models/image/supersimplenet/torch_model.py)
- **AnomalyDINO** uses a DINO encoder, a matmul-based distance ("cdist is not for half precision, but matmul is") and `torch.topk`. **MH-PatchCore** defaults to `wide_resnet50_2.tv2_in1k` with chunked `topk` nearest neighbours. **RAD** defaults to `vit_base_patch16_dinov3` — [anomaly_dino/torch_model.py](https://github.com/open-edge-platform/anomalib/blob/lib/v2.7.0/src/anomalib/models/image/anomaly_dino/torch_model.py), [mh_patchcore/torch_model.py](https://github.com/open-edge-platform/anomalib/blob/lib/v2.7.0/src/anomalib/models/image/mh_patchcore/torch_model.py), [rad/lightning_model.py](https://github.com/open-edge-platform/anomalib/blob/lib/v2.7.0/src/anomalib/models/image/rad/lightning_model.py)
- **EfficientAD** defaults to `model_size=S` and needs the ImageNette dataset (`imagenet_dir="./datasets/imagenette"`) for its penalty loss — [efficient_ad/lightning_model.py:62-91](https://github.com/open-edge-platform/anomalib/blob/lib/v2.7.0/src/anomalib/models/image/efficient_ad/lightning_model.py)

**Published accuracy and speed**
- SuperSimpleNet paper, unsupervised image AUROC / pixel AUPRO (all measured on a V100S at 256x256):

  | Method | MVTec AD | VisA | Inference time |
  |---|---|---|---|
  | SuperSimpleNet | 98.4 / 91.1 | 93.4 / 87.4 | 9.3 ms |
  | PatchCore | 98.7 / 92.7 | 94.3 / 79.7 | 54.1 ms |
  | EfficientAD | 99.1 / 93.5 | 98.1 / 94.0 | 6.0 ms |
  | SimpleNet | 97.6 / 90.5 | 91.2 / 88.0 | 22.4 ms |

  SuperSimpleNet can also use labelled anomalous images when available — [SuperSimpleNet arXiv 2408.03143](https://arxiv.org/html/2408.03143v2)
- EfficientAD reports 2 ms latency and 600 img/s on a modern GPU. It is a student-teacher model trained on normal images only, plus an autoencoder for logical anomalies — [EfficientAD arXiv 2303.14535](https://arxiv.org/abs/2303.14535)
- AnomalyDINO is a training-free DINOv2 patch kNN method. It improves 1-shot MVTec AD AUROC from 93.1% to 96.6% — [AnomalyDINO arXiv 2405.14529](https://arxiv.org/abs/2405.14529)

**Robustness benchmarks**
- RAD robustness benchmark (Cheng et al., CASE 2024; viewpoint, uneven illumination and blur; foreign objects on a platform):
  - The paper concludes that "memory-bank-based methods have better robustness for views and illuminations", crediting PatchCore coreset sampling.
  - "Light spots caused by uneven illuminations are easily identified as anomalies" by flow and distillation methods (CDO is the exception, helped by synthetic anomalies).
  - Mean image AUROC: WinCLIP 98.6, RD++ 98.3, PatchCore 97.5. PatchCore pixel AUPRO is 79.8, against 94.4 for CDO.
  - PaDiM, EfficientAD and SimpleNet were not tested.
  - [RAD arXiv 2406.07176](https://arxiv.org/html/2406.07176v2)
- A different "RAD" (Zhou et al., arXiv 2410.00713) is a robot multi-view, pose-agnostic benchmark: 13 categories, 68 viewpoints, uncontrolled lighting. Its finding is that "mature 2D feature-embedding methods consistently outperform recent 3D and VLM-based approaches at the image level" — [arXiv 2410.00713](https://arxiv.org/html/2410.00713v3). Do not confuse this paper with Anomalib's new RAD *model*.

**Closest prior example**
- Intel's Dobot tutorial (2023) is the closest example to this task: coloured cubes, with black stickers as defects, trained on normal-only images and exported to OpenVINO — [Intel article](https://www.intel.com/content/www/us/en/developer/articles/training/defect-detection-with-anomalib.html), [OpenVINO Medium lab](https://medium.com/openvino-toolkit/hands-on-lab-how-to-perform-automated-defect-detection-using-anomalib-5c1cfed666b4)

### Inferences
**Defect type**
- A hand-drawn cross is a local, high-contrast texture anomaly on a smooth plastic surface. This is the easy case for patch-feature methods. Logical-anomaly capacity (EfficientAD's autoencoder) is not needed.

**Colour variety**
- Colour variety is the main risk. A memory bank (PatchCore) naturally stores every colour seen. PaDiM fits a single Gaussian per patch position, so a multi-colour training set inflates its covariance and lowers sensitivity. If colours are many, train one PaDiM per colour, or prefer PatchCore.
- Ensure every colour and brick type is in the 50-200 training images.

**PatchCore memory bank on the NPU**
- Estimated size at 256x256 with wide_resnet50 layer2+3 (32x32 patches, 1536-d) and 200 images at 10% coreset: about 20k x 1536, roughly 60 MB in FP16, held as a graph constant.
- Estimated compute: about 31 GFLOP per frame for the matmul. With resnet18 layers 2+3 (384-d) this drops about 4x.
- I expect it to compile because the ops are MatMul/ReduceMin/TopK, but compile success of TopK on the NPU is unverified (see Gaps). Use `num_neighbors=1` at inference, because the code then uses `min` rather than `topk`.

**PaDiM on the NPU**
- With resnet18 at 256, layer1 gives 64x64 = 4096 patches. With `n_features=100`, `inv_covariance` holds 4096x100x100 = 41M values (about 82 MB in FP16). This is a batched MatMul, likely NPU-friendly but memory-heavy.

**Ordering for this task (my judgement, not benchmarked)**
1. PatchCore-resnet18 or PaDiM-resnet18 for speed. Both train in under a minute because there is no gradient training.
2. PatchCore-WRN50 for accuracy.
3. SuperSimpleNet once about 10-20 crosses can be labelled.
4. EfficientAD-S if more data and training time are available.

### Gaps
- No public per-model NPU compile results exist for Anomalib models. I found no Anomalib GitHub issues mentioning NPU: `gh search issues` returned no relevant hits. Compile success must be tested per model.
- No benchmark covers hand-drawn marks or Lego/Duplo bricks specifically.
- Accuracy of the v2.7 additions (MH-PatchCore, RAD, FoundAD) was not verified from papers.

## 2. Anomalib 2.7 export and inference API; NPU compile, caching and blobs

### Takeaway
Use `Engine.export(model, ExportType.OPENVINO, input_size=(H,W), compression_type=CompressionType.FP16 or INT8_PTQ, datamodule=dm)`. The exported IR has a **dynamic batch axis**. Before compiling on the NPU, reshape it to a static `[N,3,H,W]`. The NPU's dynamic-shape support is a preview feature, limited to bounded dimensions and the Compiler-In-Plugin path.

Compile with `CACHE_DIR` set. Anomalib's `OpenVINOInferencer` already sets `./openvino_cache`. Do not ship precompiled blobs: Intel says blobs are for development only.

### Cited Findings
**`Engine.export` signature**
```python
export(model, export_type, export_root=None, model_file_name="model",
       input_size=None, compression_type=None, datamodule=None,
       metric=None, max_drop=0.01,
       ov_args=None,  # deprecated
       ov_kwargs=None, onnx_kwargs=None, ckpt_path=None)
```
`input_size` is described as "A static input shape for the model" — [engine/engine.py:809-849](https://github.com/open-edge-platform/anomalib/blob/lib/v2.7.0/src/anomalib/engine/engine.py)

**`ExportType` and `CompressionType` values**
- `ExportType` is ONNX, OPENVINO or TORCH.
- `CompressionType` is FP16, INT8, INT8_PTQ or INT8_ACQ.
- [deploy/export.py](https://github.com/open-edge-platform/anomalib/blob/lib/v2.7.0/src/anomalib/deploy/export.py)

**How `to_openvino` works**
- It first exports ONNX through `torch.onnx.export(..., dynamo=True, opset_version=14 default)`.
- It then runs `ov.convert_model(onnx, **ov_kwargs)` and saves with `ov.save_model(model, path, compress_to_fp16=(compression_type==FP16))`.
- Without compression the IR is saved as FP32 (`compress_to_fp16=False`).
- [export_mixin/mixin.py](https://github.com/open-edge-platform/anomalib/blob/lib/v2.7.0/src/anomalib/models/components/base/export_mixin/mixin.py)

**Compression options**
- **INT8** = `nncf.compress_weights(model)` (weights only).
- **INT8_PTQ** = `nncf.quantize(model, nncf.Dataset(datamodule.val_dataloader(), lambda x: x["image"]))`. It logs a warning if there are fewer than 300 images.
- **INT8_ACQ** = accuracy-controlled quantization with `max_drop`. The default metric is image-level F1.
- Same file — [mixin.py](https://github.com/open-edge-platform/anomalib/blob/lib/v2.7.0/src/anomalib/models/components/base/export_mixin/mixin.py)

**Default dynamic axes**
- With `input_size` set, only `{0: "batch_size"}` is dynamic on the input and on every output.
- Without `input_size`, H and W are also dynamic.
- You can override this with `onnx_kwargs={"dynamic_shapes": ...}` or `dynamic_axes`.
- [export_mixin/utils.py get_default_dynamic_axes](https://github.com/open-edge-platform/anomalib/blob/lib/v2.7.0/src/anomalib/models/components/base/export_mixin/utils.py)

**What the exported graph contains**
- The exported forward is `pre_processor → model → post_processor` (`AnomalibModule.forward`), so the IR includes the following:
  - The resize and normalize transform. It is made exportable by `get_exportable_transform`, which disables antialiasing in `Resize` and converts `CenterCrop` to `ExportableCenterCrop`.
  - Min-max score normalisation and thresholding to `pred_label`/`pred_mask`.
- Sources: [anomalib_module.py:246-271](https://github.com/open-edge-platform/anomalib/blob/lib/v2.7.0/src/anomalib/models/components/base/anomalib_module.py), [pre_processing/utils/transform.py](https://github.com/open-edge-platform/anomalib/blob/lib/v2.7.0/src/anomalib/pre_processing/utils/transform.py), [post_processor.py forward](https://github.com/open-edge-platform/anomalib/blob/lib/v2.7.0/src/anomalib/post_processing/post_processor.py)

**`OpenVINOInferencer`**
- Constructor is `OpenVINOInferencer(path, device="AUTO", config=None)`. Device can be AUTO, CPU, GPU or NPU, and the value is passed straight to OpenVINO.
- It always calls `core.set_property({"CACHE_DIR": "./openvino_cache"})` (relative to the working directory), then `core.compile_model(model, device, config)`.
- Python pre-processing casts to float32, divides by 255 if max > 1, adds a batch dimension and transposes HWC to CHW. It does **not** resize.
- Outputs come back as a `NumpyImageBatch` with fields named after the model outputs (`pred_score`, `pred_label`, `anomaly_map`, `pred_mask`).
- [openvino_inferencer.py](https://github.com/open-edge-platform/anomalib/blob/lib/v2.7.0/src/anomalib/deploy/inferencers/openvino_inferencer.py)

**NPU plugin compilers**
- Since OpenVINO 2026.1, Compiler-In-Plugin is the preferred compiler. The compiler ships with OpenVINO rather than the driver. Override with `ov::intel_npu::compiler_type`.
- The plugin compiler "may not be compatible with driver versions lower than v2565".
- [OpenVINO NPU device doc source](https://github.com/openvinotoolkit/openvino/blob/master/docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.rst)

**NPU precision**
- Inference types are F32 and F16, plus U8 for quantized models (INT8 or mixed FP16-INT8).
- "Computation precision for the HW is FP16". FP32 IRs therefore run in FP16 anyway.
- Same doc — [npu-device.rst](https://github.com/openvinotoolkit/openvino/blob/master/docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.rst)

**NPU caching**
- UMD model caching in the driver is on by default (`NPU_BYPASS_UMD_CACHE=false`).
- Setting `ov::cache_dir` makes OpenVINO's cache take over. On a cache hit, the plugin imports instead of recompiling.
- Same doc — [npu-device.rst](https://github.com/openvinotoolkit/openvino/blob/master/docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.rst)

**NPU compile-tuning properties**
- `NPU_COMPILATION_MODE_PARAMS` accepts `optimization-level=0|1|2`:
  - 0 is faster compile.
  - 1 is the default.
  - 2 favours runtime performance.
- It also accepts `performance-hint-override=latency`.
- `NPU_TURBO` / `ov::intel_npu::turbo(true)` keeps a higher NPU frequency. The doc warns it can raise compile time, memory and power, so it is not recommended for sustained loads.
- `ov::intel_npu::tiles` and `max_tiles` control tile usage.
- Same doc — [npu-device.rst](https://github.com/openvinotoolkit/openvino/blob/master/docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.rst)

**NPU dynamic shapes and blobs**
- Dynamic shapes on the NPU are a preview feature from NPU40XX (Lunar Lake) onward:
  - They need Linux NPU driver 1.30+ and Compiler-In-Plugin.
  - Only bounded shapes are supported.
  - They have been validated mainly on super-resolution vision models.
- "Offline compilation and blob import is supported only for development purposes. Pre-compiled models (blobs) are not recommended to be used in production. Blob compatibility across different OpenVINO versions is not guaranteed."
- Same doc — [npu-device.rst](https://github.com/openvinotoolkit/openvino/blob/master/docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.rst)

### Inferences
Recommended recipe (not tested on this hardware):
```python
from anomalib.engine import Engine
from anomalib.deploy import ExportType, CompressionType
xml = engine.export(model, ExportType.OPENVINO, export_root="exp", input_size=(256, 256),
                    compression_type=CompressionType.FP16)   # or INT8_PTQ with datamodule=dm
import openvino as ov
core = ov.Core(); core.set_property({"CACHE_DIR": "/var/cache/brick_ov"})  # absolute path, not cwd
m = core.read_model(xml)
m.reshape({m.input(0): ov.PartialShape([3, 3, 256, 256])})   # static batch=3 for the 3 frames
cm = core.compile_model(m, "NPU", {"PERFORMANCE_HINT": "LATENCY"})
```

**Precision**
- FP16 export is the safe default because NPU compute is FP16 anyway.
- INT8_PTQ is normally the speed win on NPUs. However, PTQ calibrates on `val_dataloader()`, and the backbone here already takes about 0.9 ms. The whole-graph gain may be small, and the quantization noise on distance/Mahalanobis heads is untested.
- If you use INT8, quantize only the backbone, for example with an NNCF `ignored_scope` covering the distance/blur subgraph.
- Validate with INT8_ACQ.
- Note that Anomalib's PTQ warns below 300 images, and this task has 50-200.

**Compile time and caching**
- Compile time is paid once per shape and config with `CACHE_DIR`. Warm the cache at startup by compiling before the robot loop.
- Do not rely on `OpenVINOInferencer`'s relative `./openvino_cache`.
- I would skip `OpenVINOInferencer` in the hot loop. Its float32 cast and HWC→CHW transpose are done in NumPy per call, and it only exposes output(0) as `output_blob`.

### Gaps
- I found no measured NPU compile times for Anomalib models on Panther Lake (NPU5). These must be measured.
- The Panther Lake NPU platform ID and the 2026.4-specific release notes were not verified, and I could not confirm the docs platform table entry for Panther Lake.
- `CompressionType.INT8_PTQ` behaviour on the NPU with the PatchCore TopK/ReduceMin head is unverified.

## 3. Pre- and post-processing optimisation

### Takeaway
Resize and normalise are already inside the Anomalib IR. Feed the raw u8 ROI crop by using `ov.preprocess.PrePostProcessor` to declare a u8 NHWC input and convert inside the graph. Keep the anomaly map at model resolution (for example 256x256 or smaller): threshold, find the bounding box and connected component there, then scale box coordinates to the camera frame.

Batch the 3 frames as one static N=3 request. The NPU plugin internally runs batch-1 command lists, so batching saves call overhead rather than compute.

### Cited Findings
- The Anomalib export already contains `Resize` (no antialias) and `Normalize` in the graph, plus min-max normalisation and thresholds (`pred_mask`) — [pre_processor.py](https://github.com/open-edge-platform/anomalib/blob/lib/v2.7.0/src/anomalib/pre_processing/pre_processor.py), [post_processor.py](https://github.com/open-edge-platform/anomalib/blob/lib/v2.7.0/src/anomalib/post_processing/post_processor.py)
- PatchCore's anomaly map interpolates patch scores to full `image_size`, then applies a 33x33 Gaussian blur (sigma=4) as a depthwise conv at that resolution — [patchcore/anomaly_map.py](https://github.com/open-edge-platform/anomalib/blob/lib/v2.7.0/src/anomalib/models/image/patchcore/anomaly_map.py)
- MH-PatchCore uses bilinear interpolation and "valid"-padded blur — [mh_patchcore/anomaly_map.py](https://github.com/open-edge-platform/anomalib/blob/lib/v2.7.0/src/anomalib/models/image/mh_patchcore/anomaly_map.py)
- NPU plugin batching:
  - If the batch is on axis 0 and every input and output shares it, the plugin first compiles with batch forced to 1. It then runs one command list per item inside a single infer request, which is transparent to the app.
  - Performance "may be lower compared to regular batching".
  - The doc says the order will change in future (native batch first).
  - [batching-on-npu-plugin.rst](https://github.com/openvinotoolkit/openvino/blob/master/docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/npu-device/batching-on-npu-plugin.rst)
- `OpenVINOInferencer.pre_process` does the float32 cast, /255 and transpose in NumPy on the CPU each call — [openvino_inferencer.py](https://github.com/open-edge-platform/anomalib/blob/lib/v2.7.0/src/anomalib/deploy/inferencers/openvino_inferencer.py)

### Inferences
**Fold the input conversion into the graph**
```python
from openvino.preprocess import PrePostProcessor, ColorFormat
ppp = PrePostProcessor(m)
ppp.input().tensor().set_element_type(ov.Type.u8).set_layout(ov.Layout("NHWC")).set_color_format(ColorFormat.BGR)
ppp.input().preprocess().convert_element_type(ov.Type.f32).convert_color(ColorFormat.RGB).scale(255.0)
ppp.input().model().set_layout(ov.Layout("NCHW"))
m = ppp.build()
```
- This removes the per-frame NumPy cast and transpose.
- Feed a fixed-size ROI crop around the brick, since the arm puts it at a known pose.
- Export with `input_size` equal to the crop size so the in-graph Resize runs on the NPU, or crop/resize on the CPU with cv2 INTER_AREA. Antialiasing is disabled in the graph, which matters if you downscale a lot.

**Avoid full-resolution upsampling**
- Use input 224-256.
- Take the box from the 256 map, or better from the pre-upsample patch grid (32x32), then scale coordinates.
- To get the low-resolution map, cut the IR before Interpolate with an `ov.Model` built from the internal node outputs, or subclass the AnomalyMapGenerator to skip upsampling before export. Then blur with a small kernel on the CPU (cv2 on 32x32 costs microseconds).
- These are engineering suggestions, not Anomalib options.

**Bounding box**
- Use `pred_mask`, or threshold `anomaly_map` with the pixel threshold.
- Then run `cv2.connectedComponentsWithStats` and keep the largest component whose area exceeds a minimum. This is about 0.1 ms at 256².

**Three-frame aggregation**
- Use one N=3 infer request.
- Image score = median or max of the 3 `pred_score`s. The uncertainty band = (min, max), or ± the standard deviation of the 3 scores.
- Label "uncertain" if the band straddles the threshold.

**Async alternative**
- Use `AsyncInferQueue(cm, 3)` with batch-1 so frame 1 is processed while frames 2-3 are captured. This pipelines inference with capture, which batching cannot do.

### Gaps
- I did not find any measured latency for the PrePostProcessor, in-graph resize, or blur on NPU5. The relative cost of a 33x33 depthwise conv at 256² on the NPU is unknown.

## 4. Threshold calibration with few defects; false positives; image- vs pixel-level; illumination

### Takeaway
Anomalib's adaptive F1 threshold needs anomalous validation images. With a normal-only validation set it falls back to the highest normal score, which is a poor threshold.

Collect 10-30 cross-marked bricks across colours and set the image threshold yourself, for example as the midpoint between the 99th-percentile normal score and the minimum defect score, or via `image_sensitivity`. Use pixel-level maps only for localisation and the gating of gripper-region false positives.

Memory-bank methods are the most illumination- and viewpoint-robust in the RAD benchmark. Light spots are the main false-positive source.

### Cited Findings
- Thresholds are `F1AdaptiveThreshold` metrics computed on validation batches. Normalisation is min-max, with the normalised threshold set by `image_sensitivity`/`pixel_sensitivity` (default 0.5). `enable_threshold_matching` reuses the image threshold for pixels when no masks are available — [post_processor.py:63-193](https://github.com/open-edge-platform/anomalib/blob/lib/v2.7.0/src/anomalib/post_processing/post_processor.py)
- If the validation set has no anomalous images, a warning is logged and the threshold falls back to "the highest anomaly score observed in the normal validation images" — [f1_adaptive_threshold.py:166-179](https://github.com/open-edge-platform/anomalib/blob/lib/v2.7.0/src/anomalib/metrics/threshold/f1_adaptive_threshold.py)
- RAD robustness findings:
  - Uneven-illumination light spots "are easily identified as anomalies". Memory-bank methods (PatchCore, GCPF) are more robust to view and illumination changes, and synthetic-anomaly methods (CDO) also help.
  - Blur lowers anomaly scores and "increases the model's uncertainty".
  - All methods produce false positives from misaligned background structures.
  - [RAD arXiv 2406.07176](https://arxiv.org/html/2406.07176v2)
- On a robot multi-view benchmark, 2D feature-embedding methods beat 3D and VLM methods at the image level, and the gap narrows at pixel level. Reflective surfaces and sparse viewpoint coverage limit methods — [arXiv 2410.00713](https://arxiv.org/html/2410.00713v3)
- SuperSimpleNet supports mixed supervision, so a few labelled defects can be used in training — [arXiv 2408.03143](https://arxiv.org/html/2408.03143v2)

### Inferences
**Calibration procedure**
1. Hold out about 20% of the normal images.
2. Add 10-30 crosses across colours.
3. Run `engine.validate`/`test` so the F1 threshold uses real defects. Alternatively, compute the threshold on raw scores yourself, then disable Anomalib normalisation (`PostProcessor(enable_normalization=False)`) and apply your own threshold.
4. Report a three-way label: perfect, defective, or uncertain within ±margin, with the margin taken from the score spread across the 3 frames and the calibration spread.

**False positives from gripper, viewpoint and lighting**
- The gripper fingers appear in every image. If the training images include the gripper at the same pose, PatchCore learns it as normal.
- Mask a fixed gripper ROI out of the anomaly map before taking the max or the box.
- Prefer a tight crop around the brick.
- Train with the same lighting and include natural pose jitter.
- Avoid augmentations that change colour, because colour is part of "normal".

**Image-level vs pixel-level**
- Use the image score (the PatchCore max patch distance) for the label.
- Use the pixel map only for the box. Optionally require the max-scoring component's area to exceed N pixels before calling a defect, to suppress specular-spot false positives.

### Gaps
- There are no published results for hand-drawn marks on glossy, coloured toy plastic. The specular-reflection false-positive rate on Lego must be measured.

## 5. Concurrent NPU inference while the iGPU runs the VLA

### Takeaway
The NPU is a separate engine, so the detector does not take Xe compute from the VLA. However, the CPU, GPU and NPU share the SoC power budget and system memory, and Intel's own docs show offloading to the NPU when the GPU is busy. Use async infer requests and keep the detector on the NPU only. I found no published contention measurements for NPU and GPU running simultaneously; measure p50/p99 detector latency with the VLA running.

### Cited Findings
- Intel's DL Streamer heterogeneous guide says CPU, GPU and NPU are on one die and share a common SoC power (TDP) budget. When the GPU is heavily loaded, adding more inference to it causes competition for compute, which motivates offloading to the NPU — [Intel Open Edge heterogeneous systems guide](https://docs.openedgeplatform.intel.com/2026.1/edge-ai-libraries/dlstreamer/dev_guide/heterogeneous_systems_guide.html)
- The NPU's on-chip CMX SRAM and DDR system memory are connected by DMA, so traffic to system memory is shared with the rest of the SoC — [NPU-GPU co-inference article (third-party blog)](https://jonathanding.github.io/llm-learning/en/articles/npu-gpu-co-inference)
- The NPU turbo property raises NPU frequency within platform TDP limits but is not recommended for sustained workloads — [npu-device.rst](https://github.com/openvinotoolkit/openvino/blob/master/docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.rst)
- The NPU plugin supports `ov::hint::model_priority` and `ov::hint::enable_cpu_pinning` — [npu-device.rst](https://github.com/openvinotoolkit/openvino/blob/master/docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.rst)
- The user's own measurement for a ResNet18 backbone at 256x256, batch 1, was:

  | Device | Latency |
  |---|---|
  | NPU | 0.88 ms |
  | GPU | 0.59 ms |
  | CPU | 12.6 ms |

### Inferences
**Recommended setup**
- Compile with explicit `"NPU"`, not AUTO, because AUTO may pick the GPU or start on the CPU.
- Use `AsyncInferQueue` or `start_async` so the Python thread is not blocked. With about 1 ms of inference, the cost is dominated by Python overhead and copies, so preallocate input tensors (`ov.Tensor` over numpy buffers) and reuse infer requests.

**Expected contention**
- The VLA mostly stresses GPU compute and memory bandwidth. The NPU detector at about 1-3 ms per 3-frame batch adds little bandwidth (about 60-160 MB of constants if PatchCore/PaDiM heads are on the NPU, read every inference).
- A small backbone plus a compact head keeps the NPU's DDR traffic low. This favours resnet18 and a small coreset.

**Fallback**
- If the NPU fails to compile the head (TopK etc.), split the graph: backbone on the NPU, kNN or Mahalanobis on the CPU with NumPy/OpenVINO-CPU. With 20k x 384 resnet18 features this is a few ms on the CPU. HETERO:NPU,CPU is only "partially supported" for the NPU ([npu-device.rst](https://github.com/openvinotoolkit/openvino/blob/master/docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.rst)), so a manual split is safer.

### Gaps
- I found no measured latency numbers for the NPU under concurrent heavy iGPU load on Lunar Lake or Panther Lake. These need measuring by the benchmark researcher.
- The Panther Lake NPU5 memory-bandwidth share is undocumented in the sources found.
