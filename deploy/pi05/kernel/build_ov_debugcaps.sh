#!/usr/bin/env bash
# Build OpenVINO 2026.4.1 (same tag/commit as the release wheel in the Studio env: e213a147257) with
# ENABLE_DEBUG_CAPS=ON, GPU + CPU plugins (CPU is needed: physicalai runs the tokenizer IR on CPU), the IR
# frontend only, and Python bindings for ~/inference/.venv-ovdbg. No sudo needed.
#
#   bash build_ov_debugcaps.sh            # configure + build + install (run under flock, see README)
#   STAGE=configure bash build_ov_debugcaps.sh
#
# Install prefix: ~/inference/ovbuild/install. The venv sees it through a .pth (python) + LD_LIBRARY_PATH (env.sh).
set -euo pipefail
ROOT=${ROOT:-$HOME/inference/ovbuild}
VENV=${VENV:-$HOME/inference/.venv-ovdbg}
JOBS=${JOBS:-8}
STAGE=${STAGE:-all}
SRC=$ROOT/src BUILD=$ROOT/build PREFIX=$ROOT/install
export PATH=$VENV/bin:$PATH

if [[ ! -d $SRC ]]; then
  git clone --depth 1 --branch 2026.4.1 https://github.com/openvinotoolkit/openvino.git "$SRC"
  (cd "$SRC" && git submodule update --init --recursive --depth 1 --jobs 4)
fi
# GeGLU (Gemma) support in FuseGatedMLP, opt-in at runtime with OV_GPU_GATED_MLP_GELU=1
"$VENV/bin/python" "$(dirname "$0")/patch_gated_mlp_gelu.py" "$SRC"

if [[ $STAGE == all || $STAGE == configure ]]; then
  cmake -S "$SRC" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF \
    -DENABLE_DEBUG_CAPS=ON \
    -DENABLE_INTEL_GPU=ON -DENABLE_ONEDNN_FOR_GPU=ON \
    -DENABLE_INTEL_CPU=ON -DENABLE_INTEL_NPU=OFF \
    -DENABLE_AUTO=ON -DENABLE_HETERO=OFF -DENABLE_AUTO_BATCH=OFF -DENABLE_MULTI=OFF \
    -DENABLE_OV_IR_FRONTEND=ON \
    -DENABLE_OV_ONNX_FRONTEND=OFF -DENABLE_OV_PADDLE_FRONTEND=OFF -DENABLE_OV_TF_FRONTEND=OFF \
    -DENABLE_OV_TF_LITE_FRONTEND=OFF -DENABLE_OV_PYTORCH_FRONTEND=OFF -DENABLE_OV_JAX_FRONTEND=OFF \
    -DENABLE_TESTS=OFF -DENABLE_FUNCTIONAL_TESTS=OFF -DENABLE_SAMPLES=ON \
    -DENABLE_PYTHON=ON -DENABLE_WHEEL=OFF -DPython3_EXECUTABLE="$VENV/bin/python" \
    -DENABLE_CLANG_FORMAT=OFF -DENABLE_NCC_STYLE=OFF -DENABLE_CPPLINT=OFF \
    -DENABLE_SYSTEM_TBB=OFF -DENABLE_JS=OFF
fi
if [[ $STAGE == all || $STAGE == build ]]; then
  cmake --build "$BUILD" --parallel "$JOBS"
  cmake --install "$BUILD"
  # Python package: the install puts it under $PREFIX/python; expose it to the venv (ahead of the Studio pth below)
  SP=$("$VENV/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
  echo "$PREFIX/python" > "$SP/00_ov_debugcaps.pth"
  # physicalai, openvino_tokenizers, torch, ... from the Studio env (read-only; its openvino is shadowed by the line above)
  echo "$HOME/inference/physical-ai-studio/library/.venv/lib/python3.12/site-packages" > "$SP/zz_studio_site.pth"
  cat > "$ROOT/env.sh" <<ENV
export LD_LIBRARY_PATH=$PREFIX/runtime/lib/intel64:$PREFIX/runtime/3rdparty/tbb/lib:\${LD_LIBRARY_PATH:-}
export PATH=$PREFIX/samples_bin:\$PATH
ENV
fi
echo BUILD_DONE
