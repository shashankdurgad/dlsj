"""Physical AI Studio as the model registry for the run-time loop.

The loop selects its policy by **Studio model ID**. This module resolves that ID to an OpenVINO export directory
that `physicalai.inference.InferenceModel` can load:

* Studio's REST API (`GET /api/models/{id}`) gives the record: name, policy, project, training dataset, and the
  registered export (backend, total size, file count).
* The export files live in a local mirror, `~/inference/studio_models/<id>/<backend>/`, next to a
  `studio_model.json` record. The mirror is filled either by `seed` (hard links to the files that were imported
  into Studio, after checking they are byte-identical to Studio's copy) or by downloading the export through
  Studio's own `GET /api/models/{id}/exports/{backend}/download`.
* `resolve` checks the mirror against the Studio record (file count and total bytes) every time. If Studio is not
  reachable it falls back to the saved record, says so, and still checks the files against it, so a Studio
  restart never stops the robot.

    python -m dlsj.deploy.studio_registry resolve <model-id>      # prints the export dir (logs on stderr)
    python -m dlsj.deploy.studio_registry show <model-id>         # the record as JSON
    python -m dlsj.deploy.studio_registry seed <model-id> <dir> [--studio-sums sums.txt]
    python -m dlsj.deploy.studio_registry list <project-id>

Only the standard library is used, so it runs in any of the XPS venvs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

STUDIO_URL = os.environ.get("STUDIO_URL", "http://localhost:7860")
ROOT = Path(os.environ.get("STUDIO_MODELS", "~/inference/studio_models")).expanduser()
RECORD = "studio_model.json"


class StudioError(RuntimeError):
    pass


def _log(msg: str) -> None:
    print(f"[studio] {msg}", file=sys.stderr, flush=True)


class Studio:
    """Minimal client for the Studio backend API."""

    def __init__(self, url: str = STUDIO_URL, timeout: float = 5.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def get(self, path: str) -> dict:
        try:
            with urllib.request.urlopen(self.url + path, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise StudioError(f"GET {path}: HTTP {e.code}") from e

    def model(self, model_id: str) -> dict:
        return self.get(f"/api/models/{model_id}")

    def dataset(self, dataset_id: str) -> dict:
        return self.get(f"/api/dataset/{dataset_id}")

    def project(self, project_id: str) -> dict:
        return self.get(f"/api/projects/{project_id}")

    def models(self, project_id: str) -> list[dict]:
        d = self.get(f"/api/projects/{project_id}/models")
        return d if isinstance(d, list) else d.get("data", d.get("models", []))

    def download_export(self, model_id: str, backend: str, dest: Path) -> None:
        """Stream Studio's export archive to `dest` (a zip)."""
        req = f"{self.url}/api/models/{model_id}/exports/{backend}/download"
        with urllib.request.urlopen(req, timeout=max(self.timeout, 60)) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f, 16 << 20)


def export_dir(model_id: str, backend: str = "openvino", root: Path = ROOT) -> Path:
    return root / model_id / backend


def files_of(d: Path) -> dict[str, int]:
    return {p.name: p.stat().st_size for p in sorted(d.iterdir()) if p.is_file()}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def registered_export(remote: dict, backend: str) -> dict:
    for e in remote.get("exports", []):
        if e.get("type") == backend:
            return e
    raise StudioError(f"model {remote['model']['id']} has no {backend} export in Studio "
                      f"(available: {remote['model'].get('available_backends')})")


def check_files(d: Path, export: dict) -> None:
    """The mirror must hold exactly the registered export: same file count and total bytes."""
    if not (d / "manifest.json").is_file():
        raise StudioError(f"{d}: no manifest.json")
    files = files_of(d)
    count, size = len(files), sum(files.values())
    if count != export["file_count"] or size != export["size_bytes"]:
        raise StudioError(f"{d}: {count} files / {size} bytes, Studio has {export['file_count']} files / "
                          f"{export['size_bytes']} bytes")


def make_record(remote: dict, backend: str, studio: Studio, files: dict[str, int], **extra) -> dict:
    m, e = remote["model"], registered_export(remote, backend)
    rec = {"model_id": m["id"], "name": m["name"], "policy": m["policy"], "version": m.get("version"),
           "project_id": m["project_id"], "dataset_id": m["dataset_id"], "backend": backend,
           "export": {"size_bytes": e["size_bytes"], "file_count": e["file_count"],
                      "exported_at": e.get("exported_at")},
           "files": files, "studio_url": studio.url, "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for key, fetch in (("project", studio.project), ("dataset", studio.dataset)):
        try:
            rec[f"{key}_name"] = fetch(m[f"{key}_id"]).get("name")
        except (OSError, StudioError, KeyError):
            pass
    rec.update(extra)
    return rec


def read_record(model_id: str, root: Path = ROOT) -> dict | None:
    p = root / model_id / RECORD
    return json.loads(p.read_text()) if p.is_file() else None


def record_for_export(export: str | Path) -> dict | None:
    """The Studio record next to an export dir (`<root>/<id>/<backend>`), or None for a non-Studio path."""
    p = Path(export).expanduser().resolve().parent / RECORD
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def write_record(rec: dict, root: Path = ROOT) -> None:
    p = root / rec["model_id"] / RECORD
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=2))
    tmp.replace(p)


def _flatten_zip(zpath: Path, dest: Path) -> None:
    """Extract Studio's archive (weights under exports/<backend>/) so dest holds the manifest and its files."""
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp, zipfile.ZipFile(zpath) as z:
        z.extractall(tmp)
        manifests = list(Path(tmp).rglob("manifest.json"))
        if len(manifests) != 1:
            raise StudioError(f"expected one manifest.json in the Studio archive, found {len(manifests)}")
        shutil.rmtree(dest, ignore_errors=True)
        shutil.move(str(manifests[0].parent), dest)


def download(model_id: str, backend: str, studio: Studio, root: Path = ROOT) -> Path:
    d = export_dir(model_id, backend, root)
    d.parent.mkdir(parents=True, exist_ok=True)
    zpath = d.parent / f"{backend}.zip.partial"
    _log(f"downloading the {backend} export of {model_id} from {studio.url} ...")
    try:
        studio.download_export(model_id, backend, zpath)
        _flatten_zip(zpath, d)
    finally:
        zpath.unlink(missing_ok=True)
    return d


def resolve(model_id: str, backend: str = "openvino", studio: Studio | None = None, root: Path = ROOT,
            allow_download: bool = True) -> tuple[Path, dict]:
    """Studio model ID -> (verified export dir, record). Works offline from the saved record."""
    studio = studio or Studio()
    d = export_dir(model_id, backend, root)
    saved = read_record(model_id, root)
    try:
        remote = studio.model(model_id)
    except (OSError, StudioError) as e:
        if saved is None or not d.is_dir():
            raise StudioError(f"Studio at {studio.url} unreachable ({e}) and no local copy of {model_id}") from e
        check_files(d, saved["export"])
        _log(f"Studio unreachable ({e}); using the saved record of '{saved['name']}' (files checked against it)")
        return d, {**saved, "online": False}
    export = registered_export(remote, backend)
    try:
        check_files(d, export)
    except StudioError as e:
        if not allow_download:
            raise
        _log(f"local copy not usable ({e})")
        download(model_id, backend, studio, root)
        check_files(d, export)
    extra = {k: saved[k] for k in ("seeded_from", "sha256_verified", "lineage") if saved and k in saved}
    rec = make_record(remote, backend, studio, files_of(d), **extra)
    write_record(rec, root)
    _log(f"'{rec['name']}' ({rec['policy']}, Studio model {model_id}) -> {d}")
    return d, {**rec, "online": True}


def seed(model_id: str, src: Path, backend: str = "openvino", studio: Studio | None = None, root: Path = ROOT,
         studio_sums: Path | None = None, lineage: dict | None = None) -> Path:
    """Fill the mirror with hard links to `src`, the folder that was imported into Studio.

    `studio_sums` is `sha256sum` output for Studio's own copy (run inside the container); every file must match.
    """
    studio = studio or Studio()
    remote = studio.model(model_id)
    export = registered_export(remote, backend)
    src = src.expanduser().resolve()
    check_files(src, export)
    verified = False
    if studio_sums:
        theirs = {Path(n.strip()).name: h for h, n in (ln.split(None, 1) for ln in studio_sums.read_text().splitlines()
                                                         if ln.strip())}
        ours = {p.name: sha256(p) for p in sorted(src.iterdir()) if p.is_file()}
        if ours != theirs:
            bad = sorted(k for k in set(ours) | set(theirs) if ours.get(k) != theirs.get(k))
            raise StudioError(f"{src} differs from Studio's copy: {bad}")
        verified = True
    d = export_dir(model_id, backend, root)
    tmp = d.with_name(d.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for p in sorted(src.iterdir()):
        if p.is_file():
            try:
                os.link(p, tmp / p.name)
            except OSError:
                shutil.copy2(p, tmp / p.name)
    shutil.rmtree(d, ignore_errors=True)
    tmp.rename(d)
    rec = make_record(remote, backend, studio, files_of(d), seeded_from=str(src), sha256_verified=verified,
                      **({"lineage": lineage} if lineage else {}))
    write_record(rec, root)
    _log(f"seeded {d} from {src} (sha256 {'matches Studio' if verified else 'not checked'})")
    return d


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--studio", default=STUDIO_URL)
    ap.add_argument("--root", type=Path, default=ROOT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("resolve")
    r.add_argument("model_id")
    r.add_argument("--backend", default="openvino")
    r.add_argument("--no-download", action="store_true")
    s = sub.add_parser("show")
    s.add_argument("model_id")
    s.add_argument("--backend", default="openvino")
    sd = sub.add_parser("seed")
    sd.add_argument("model_id")
    sd.add_argument("src", type=Path)
    sd.add_argument("--backend", default="openvino")
    sd.add_argument("--studio-sums", type=Path)
    sd.add_argument("--lineage", type=Path, help="JSON describing how the model was trained (kept in the record)")
    ls = sub.add_parser("list")
    ls.add_argument("project_id")
    a = ap.parse_args(argv)
    studio = Studio(a.studio)
    try:
        if a.cmd == "resolve":
            d, _ = resolve(a.model_id, a.backend, studio, a.root, allow_download=not a.no_download)
            print(d)
        elif a.cmd == "show":
            _, rec = resolve(a.model_id, a.backend, studio, a.root, allow_download=False)
            print(json.dumps(rec, indent=2))
        elif a.cmd == "seed":
            lineage = json.loads(a.lineage.read_text()) if a.lineage else None
            print(seed(a.model_id, a.src, a.backend, studio, a.root, a.studio_sums, lineage))
        elif a.cmd == "list":
            for m in studio.models(a.project_id):
                print(m["id"], m.get("policy"), m.get("name"))
    except StudioError as e:
        sys.exit(f"studio_registry: {e}")


if __name__ == "__main__":
    main()
