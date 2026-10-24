"""Nemotron vision check on a few key frames, turned into vision_checks rows.

Only key frames go to the model (each job takes about 13 s). The numbers come from
metrics.py; vision answers the questions numbers cannot: what is in frame, and
whether the result looks right.

SAMPLE_MODE=1 replays saved answers from a JSON cache, so judges need no Nightshift.
"""
import json
import os
from pathlib import Path

from . import nightshift

# Classify: what is in frame. Verify: does the result look right.
QUESTIONS = {
    "contents": "List the main objects in this frame (for example car, sky, buildings, "
                "people) and where each one sits. One short line per object.",
    "highlights": "Where is the brightest area, and does it look clipped (flat white "
                  "with no detail)?",
    "seams": "Do you see seams, halos or flicker-like edges around any object? "
             "Answer yes or no, then say where.",
}

SAMPLE_MODE = os.environ.get("SAMPLE_MODE", "0") == "1"


def _cache_key(shot_id, frame, key):
    return f"{shot_id}:f{frame}:{key}"


def check_frame(shot_id, frame, image_path, keys=("contents", "highlights"),
                cache_path="examples/vision_cache.json", step="classify"):
    """Ask the chosen questions about one frame. Returns vision_checks rows.

    Live mode calls Nightshift and adds each answer to the cache.
    Sample mode reads the cache and never calls the model.
    """
    cache_file = Path(cache_path)
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    rows = []
    for key in keys:
        ck = _cache_key(shot_id, frame, key)
        if SAMPLE_MODE:
            if ck not in cache:
                raise KeyError(f"sample mode: no saved answer for {ck} in {cache_path}")
            hit = cache[ck]
        else:
            answer, job = nightshift.vision(QUESTIONS[key], str(image_path), step=step)
            hit = {"answer": answer, "job_id": job.get("id"),
                   "model": job.get("answer_model"),
                   "receipt": job.get("meterless.receipt") or {"lanes_resolved": job.get("lanes_resolved")}}
            cache[ck] = hit
        rows.append({
            "evidence_id": f"vision:{shot_id}:f{frame}:{key}",
            "shot_id": shot_id,
            "frame": frame,
            "question": QUESTIONS[key],
            "answer": hit["answer"],
            "job_id": hit.get("job_id"),
            "model": hit.get("model"),
            "receipt": json.dumps(hit.get("receipt"), default=str),
        })
    if not SAMPLE_MODE:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache, indent=2, default=str))
    return rows
