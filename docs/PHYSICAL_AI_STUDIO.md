# Local Physical AI Studio

Studio source is tracked as a Git submodule at
`third_party/physical-ai-studio`, pinned to upstream commit `8f132416`.
Its license and Git history remain in the submodule. Python environments,
frontend dependencies, datasets, and model weights are not committed.

## Setup

Install Python 3.13, uv, Node >=24.2, npm >=11.14, and FFmpeg, then from the
repository root:

```bash
git submodule update --init third_party/physical-ai-studio
cd third_party/physical-ai-studio/application/backend
uv sync --extra cpu --no-default-groups
cd ../ui
npm ci
PUBLIC_API_BASE_URL=http://localhost:7860 npm run build
```

The CPU extra is the starting configuration for the Mac. Use the upstream
hardware setup instructions for Intel XPU or NVIDIA CUDA systems. Running the
UI does not validate model inference, simulation, or physical robot execution.

## Run

From the repository root:

```bash
./scripts/run_studio.sh
```

Open <http://localhost:7860>. The server binds to loopback. Stop it with Ctrl+C.
On macOS, Studio stores its database and artifacts under
`~/Library/Application Support/physicalai` by default. Set `STORAGE_DIR` to
use another location. The launcher applies Studio's database migrations at
startup. No Hugging Face token is needed just to open the UI.

After UI changes, rerun the build command above and refresh the browser. Backend
changes require restarting the launcher. The editable Python installation uses
this submodule's source directly.

## Making and sharing Studio changes

Create a branch inside the submodule before editing:

```bash
git -C third_party/physical-ai-studio switch -c feat/our-studio-changes
```

Studio changes are committed inside the submodule. To share them, push those
commits to a team-accessible Studio fork, update the URL in `.gitmodules` to
that fork, and commit the URL and submodule pointer in `dlsj`. The current URL
points at the official upstream, so everyone can fetch the pinned baseline.
Do not commit a parent pointer until its Studio commit is available remotely.
