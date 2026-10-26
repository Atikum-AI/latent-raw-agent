"""Where the agent finds each clip's files, and how it measures them.

Layout (agreed with Kyo, who delivers one key frame per clip):

    samples/<clip_id>/
        clip.json        {"key_frame": 48, "aces": "AP1", "source_model": "Seedance", "source_file": "<clip_id>.mp4"}
        source.png       the key frame, 8-bit
        masks/<item>.png SAM 3.1 + MatAnyone soft mattes, frame-sized, named by label
        synth.png        pixels the expand pass synthesized
        expand_hdr.exr   key frame after expand_hdr + to_aces_linear (ACEScg/AP1, linear, half float)
        balance.exr      key frame after the balance pass
        evidence.json    the measured rows, so sample mode needs no EXRs

The clips live in the Hugging Face dataset caiotbr/latent-raw-agent-samples
(python scripts/fetch_samples.py).
"""
import json
import os
from pathlib import Path

import numpy as np

from measure import metrics

SAMPLES_DIR = Path(os.environ.get("SAMPLES_DIR", "samples"))

# The precomputed output each pass leaves behind. to_aces_linear and write_exr share
# a file with the pass before them, because Kyo's passes write ACES EXRs directly.
PASS_FILES = {
    "expand_hdr": "expand_hdr.exr",
    "to_aces_linear": "expand_hdr.exr",
    "sam_items": "masks",
    "balance": "balance.exr",
    "write_exr": "balance.exr",
}

AFTER = "@balance"  # shot_id suffix for rows measured on the balanced EXR


def clip_dir(clip_id):
    return SAMPLES_DIR / clip_id


def clip_info(clip_id):
    p = clip_dir(clip_id) / "clip.json"
    info = json.loads(p.read_text()) if p.exists() else {}
    return {"key_frame": info.get("key_frame", 1), "aces": info.get("aces", "AP0"),
            "source_model": info.get("source_model"), "source_file": info.get("source_file")}


def pass_output(clip_id, pass_name):
    """Path of a pass's precomputed output, or None when it is not there."""
    p = clip_dir(clip_id) / PASS_FILES[pass_name]
    return p if p.exists() else None


def measure_clip(clip_id):
    """Measure the key frame from the clip's files. Returns evidence tables (dict of row lists)."""
    d, info = clip_dir(clip_id), clip_info(clip_id)
    frame, aces = info["key_frame"], info["aces"]
    src = metrics.load_8bit(d / "source.png")
    shape = src.shape[:2]
    items = metrics.load_item_masks(d / "masks", shape)
    synth = metrics.load_mask(d / "synth.png", shape) if (d / "synth.png").exists() else np.zeros(shape, bool)

    shots = [{"shot_id": clip_id, "source_model": info["source_model"], "source_file": info["source_file"]}]
    rows, ranges = [], []
    exr = d / "expand_hdr.exr"
    hdr = metrics.load_exr(exr)[1] if exr.exists() else None
    rows += metrics.item_rows(clip_id, frame, src, items, synth, hdr, aces)
    if hdr is not None:
        ranges.append(metrics.frame_range_row(clip_id, frame, src, hdr, synth, aces))
    if (d / "balance.exr").exists():
        after = clip_id + AFTER
        shots.append({"shot_id": after, "source_model": info["source_model"], "source_file": "balance.exr"})
        rows += metrics.item_rows(after, frame, src, items, synth, metrics.load_exr(d / "balance.exr")[1], aces)
    return {"shots": shots, "item_metrics": rows, "frame_range": ranges}


FETCH_HINT = "python scripts/fetch_samples.py"


class MissingSamples(FileNotFoundError):
    """A clip has neither saved measurements nor the files to measure."""


def load_evidence(clip_id, sample_mode):
    """Sample mode reads evidence.json. Live mode measures the files and saves evidence.json,
    unless the EXR is not there (a clean clone): then it reads evidence.json too."""
    d = clip_dir(clip_id)
    saved = d / "evidence.json"
    can_measure = (d / "source.png").exists() and (d / "expand_hdr.exr").exists()
    if sample_mode or not can_measure:
        if not saved.exists():
            raise MissingSamples(
                f"No measurements for {clip_id} (looked for {saved}).\n"
                f"Fetch the sample clips first:  {FETCH_HINT}")
        return json.loads(saved.read_text())
    tables = measure_clip(clip_id)
    saved.write_text(json.dumps(tables, indent=2, default=str))
    return tables
