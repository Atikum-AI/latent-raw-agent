"""Download the sample clips from Hugging Face into samples/.

    python scripts/fetch_samples.py
    python scripts/fetch_samples.py --dest samples --force

The dataset (caiotbr/latent-raw-agent-samples) holds one folder per clip in the
agent's layout: <clip_id>/clip.json, source.png, masks/<item>.png, synth.png,
expand_hdr.exr, balance.exr. A leading samples/ in the dataset paths is dropped, and
folders without a clip.json (tools/) are skipped.
It is public, so no token is needed; HF_TOKEN is used if set.
"""
import argparse
import os
from pathlib import Path

import requests

DATASET = os.environ.get("SAMPLES_DATASET", "caiotbr/latent-raw-agent-samples")
HF = "https://huggingface.co"
SKIP = {".gitattributes", "README.md"}


def _headers():
    return {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}


def _safe(dest, rel):
    """Keep every write inside dest, whatever the dataset path is."""
    out = (dest / rel).resolve()
    if dest.resolve() not in out.parents:
        raise ValueError(f"refusing to write outside {dest}: {rel}")
    return out


def list_files(dataset, revision):
    url = f"{HF}/api/datasets/{dataset}/tree/{revision}?recursive=true"
    files = []
    while url:
        r = requests.get(url, headers=_headers(), timeout=30)
        if r.status_code == 404:
            raise SystemExit(f"No dataset called {dataset} on Hugging Face (or it is private).")
        r.raise_for_status()
        files += [f["path"] for f in r.json() if f.get("type") == "file"]
        url = r.links.get("next", {}).get("url")  # long listings come in pages
    return files


def _rel(path):
    return path[len("samples/"):] if path.startswith("samples/") else path


def clip_folders(paths):
    """Only folders with a clip.json are clips; tools/ and the like are left out."""
    return {Path(_rel(p)).parts[0] for p in paths if Path(_rel(p)).name == "clip.json" and len(Path(_rel(p)).parts) == 2}


def local_path(dest, path, clips):
    rel = _rel(path)
    if Path(rel).parts[0] not in clips or Path(rel).name in SKIP or "/" not in rel:
        return None
    return _safe(dest, rel)


def fetch(dest="samples", force=False, dataset=DATASET, revision="main"):
    dest = Path(dest)
    got = 0
    paths = list_files(dataset, revision)
    clips = clip_folders(paths)
    for path in paths:
        out = local_path(dest, path, clips)
        if out is None:
            continue
        if out.exists() and not force:
            print(f"have {out}")
            continue
        print(f"get  {path}")
        r = requests.get(f"{HF}/datasets/{dataset}/resolve/{revision}/{path}", headers=_headers(), timeout=300)
        r.raise_for_status()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(r.content)
        got += 1
    print(f"done: {got} files into {dest}/")
    return got


def main(argv=None):
    p = argparse.ArgumentParser(prog="python scripts/fetch_samples.py")
    p.add_argument("--dest", default=os.environ.get("SAMPLES_DIR", "samples"))
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--revision", default="main")
    p.add_argument("--force", action="store_true", help="download files you already have")
    a = p.parse_args(argv)
    fetch(a.dest, a.force, a.dataset, a.revision)


if __name__ == "__main__":
    main()
