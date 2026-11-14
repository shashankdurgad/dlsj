# The digital-twin image: Isaac Lab 3.0's official image (built by build.sh on isaac-sim:6.2.0, arm64 on the GB10)
# plus Physical AI Studio's library, so Studio policies and gyms run in the same Python as the sim.
ARG BASE_IMAGE=dlsj/isaaclab-base:3.0.0
FROM ${BASE_IMAGE}
USER root
# Studio pulls pynokhwa (webcam capture, Rust + bindgen; no arm64 wheel): it needs libclang and the v4l headers to build.
RUN apt-get update && apt-get install -y --no-install-recommends clang libclang-dev libv4l-dev pkg-config \
 && rm -rf /var/lib/apt/lists/*
ARG STUDIO_REPO=https://github.com/open-edge-platform/physical-ai-studio.git
ARG STUDIO_REF=8f132416
RUN git clone -q --filter=blob:none ${STUDIO_REPO} /opt/physical-ai-studio \
 && git -C /opt/physical-ai-studio checkout -q ${STUDIO_REF}
# Keep Isaac Lab's torch / numpy / warp: Studio is installed against them, never over them.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip freeze --python ${VIRTUAL_ENV}/bin/python \
      | grep -iE '^(torch|torchvision|torchaudio|numpy|warp-lang|gymnasium|opencv-python|opencv-python-headless)==' \
      > /opt/isaaclab-pins.txt \
 && uv pip install --python ${VIRTUAL_ENV}/bin/python -c /opt/isaaclab-pins.txt -e /opt/physical-ai-studio/library \
 && chown -R isaaclab:isaaclab /opt/physical-ai-studio \
 && find ${VIRTUAL_ENV} \! -user isaaclab -exec chown isaaclab:isaaclab {} +
USER isaaclab
