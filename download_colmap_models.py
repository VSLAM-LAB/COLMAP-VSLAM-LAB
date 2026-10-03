"""
Module: VSLAM-LAB - Baselines - colmap - download_colmap_models.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Fable 5.1)
- Version: 1.0
- Created: 2026-10-03
- Updated: 2026-10-03
- License: GPLv3 License

Pre-warms COLMAP's download cache (`pixi run download-colmap-models`) so runs on nodes without
internet never stop to fetch a model: the ONNX feature extractors / matchers (ALIKED, LightGlue,
LoMa) and the FAISS vocabulary trees for sequential loop detection. COLMAP embeds every resource
it may download as a '<url>;<name>;<sha256>' string; this script reads them out of the installed
`colmap` binary (so the list always matches the installed version), downloads the missing ones
into COLMAP's own cache layout (<cache dir>/<sha256>-<name>, default ~/.cache/colmap, which the
CLI build cannot redirect, so run it with the HOME the jobs will see) and verifies the sha256.
The bf16 model variants are skipped unless --include-bf16 is given (the pipeline runs fp32).
"""

import argparse
import hashlib
import os
import re
import shutil
import sys
import urllib.request
from pathlib import Path

URI_RE = re.compile(rb'https://[^\x00;\s]+;[^\x00;\s]+;[0-9a-f]{64}')


def colmap_resources(colmap_binary: Path) -> list[tuple[str, str, str]]:
    """(url, name, sha256) of every downloadable resource embedded in the colmap binary, sorted by name."""
    data = colmap_binary.read_bytes()
    found = {m.group(0).decode('ascii') for m in URI_RE.finditer(data)}
    resources = sorted((tuple(uri.split(';')) for uri in found), key=lambda r: r[1])
    return [(url, name, sha) for url, name, sha in resources]


def default_cache_dir() -> Path:
    return Path(os.path.expanduser("~")) / ".cache" / "colmap"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, target: Path, expected_sha256: str) -> None:
    """Download url to target via a temporary file, verifying the sha256 before the rename."""
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(tmp, 'wb') as out:
        total = response.headers.get('Content-Length')
        total = int(total) if total else None
        done = 0
        for chunk in iter(lambda: response.read(1 << 20), b''):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r        {done / 2**20:8.1f} / {total / 2**20:.1f} MB", end='', flush=True)
        print()
    actual = sha256_of(tmp)
    if actual != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"sha256 mismatch for {target.name}: expected {expected_sha256}, got {actual}")
    tmp.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-download COLMAP's ONNX models and vocabulary trees into its cache")
    parser.add_argument("--cache-dir", dest="cache_dir", type=Path, default=default_cache_dir(),
                        help="COLMAP download cache (default: ~/.cache/colmap, the only location the colmap CLI reads)")
    parser.add_argument("--include-bf16", dest="include_bf16", action="store_true", help="also fetch the *_bf16 model variants")
    parser.add_argument("--list", action="store_true", help="print the resources and their cache state, download nothing")
    parser.add_argument("--colmap", type=Path, default=None, help="colmap binary to read the resource list from (default: the one on PATH)")
    args = parser.parse_args()

    colmap_binary = args.colmap or (Path(shutil.which("colmap")) if shutil.which("colmap") else None)
    if colmap_binary is None or not colmap_binary.is_file():
        print("colmap binary not found (run inside the colmap pixi environment or pass --colmap)")
        sys.exit(1)

    resources = colmap_resources(colmap_binary)
    if not args.include_bf16:
        resources = [r for r in resources if "_bf16" not in r[1]]
    print(f"colmap binary : {colmap_binary}")
    print(f"cache dir     : {args.cache_dir}")
    print(f"resources     : {len(resources)}" + ("" if args.include_bf16 else " (bf16 variants skipped, --include-bf16 to add them)"))

    missing = []
    for url, name, sha in resources:
        target = args.cache_dir / f"{sha}-{name}"
        state = "cached" if target.is_file() else "missing"
        print(f"    [{state:7s}] {name}")
        if state == "missing":
            missing.append((url, name, sha, target))
    if args.list or not missing:
        print("nothing to download" if not missing else f"{len(missing)} missing (run without --list to download)")
        return

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for i, (url, name, sha, target) in enumerate(missing, 1):
        print(f"    downloading [{i}/{len(missing)}] {name}\n        from {url}")
        try:
            download(url, target, sha)
        except Exception as e:  # keep going, report at the end
            print(f"        FAILED: {e}")
            failed.append(name)
    if failed:
        print(f"{len(failed)} download(s) failed: {failed}")
        sys.exit(1)
    print(f"done: {len(missing)} file(s) added to {args.cache_dir}")


if __name__ == "__main__":
    main()
