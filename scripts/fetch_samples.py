"""Download the sample-data-v1 GitHub Release into samples/.

    python scripts/fetch_samples.py
    python scripts/fetch_samples.py --tag sample-data-v1 --dest samples --force

Release assets can be either:
  - one file per clip file, named <clip_id>__<file>, e.g. clip01_car_street_sun__expand_hdr.exr
    -> samples/clip01_car_street_sun/expand_hdr.exr
  - a .zip of clip folders (with or without a leading samples/) -> unpacked into samples/

The repo is public, so no token is needed. GITHUB_TOKEN is used if set (higher rate limit).
"""
import argparse
import io
import os
import sys
import zipfile
from pathlib import Path

import requests

REPO = os.environ.get("SAMPLES_REPO", "Atikum-AI/latent-raw-agent")


def _headers():
    h = {"Accept": "application/vnd.github+json"}
    if os.environ.get("GITHUB_TOKEN"):
        h["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    return h


def _safe(dest, rel):
    """Keep every write inside dest, whatever the asset or zip entry is called."""
    out = (dest / rel).resolve()
    if dest.resolve() not in out.parents:
        raise ValueError(f"refusing to write outside {dest}: {rel}")
    return out


def target(dest, name):
    if "__" not in name:
        return None
    clip, file = name.split("__", 1)
    return _safe(dest, Path(clip) / file)


def unpack(dest, data):
    n = 0
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            rel = info.filename[len("samples/"):] if info.filename.startswith("samples/") else info.filename
            out = _safe(dest, rel)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(z.read(info))
            n += 1
    return n


def fetch(tag="sample-data-v1", dest="samples", force=False):
    dest = Path(dest)
    r = requests.get(f"https://api.github.com/repos/{REPO}/releases/tags/{tag}", headers=_headers(), timeout=30)
    if r.status_code == 404:
        sys.exit(f"No release called {tag} in {REPO} yet.")
    r.raise_for_status()
    got = 0
    for asset in r.json().get("assets", []):
        name, url = asset["name"], asset["browser_download_url"]
        out = None if name.endswith(".zip") else target(dest, name)
        if not name.endswith(".zip") and out is None:
            print(f"skip {name}: expected <clip_id>__<file> or a .zip")
            continue
        if out is not None and out.exists() and not force:
            print(f"have {out}")
            continue
        print(f"get  {name} ({asset.get('size', 0) // 1024} KB)")
        data = requests.get(url, headers={"Accept": "application/octet-stream"}, timeout=300).content
        if out is None:
            got += unpack(dest, data)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            got += 1
    print(f"done: {got} files into {dest}/")
    return got


def main(argv=None):
    p = argparse.ArgumentParser(prog="python scripts/fetch_samples.py")
    p.add_argument("--tag", default="sample-data-v1")
    p.add_argument("--dest", default=os.environ.get("SAMPLES_DIR", "samples"))
    p.add_argument("--force", action="store_true", help="download files you already have")
    a = p.parse_args(argv)
    fetch(a.tag, a.dest, a.force)


if __name__ == "__main__":
    main()
