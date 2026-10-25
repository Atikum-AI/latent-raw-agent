"""Ask Nemotron (through Nightshift) for a decision as JSON, and keep its receipt.

Live mode calls measure.nightshift.chat and saves each answer in a cache keyed by the
prompt. SAMPLE_MODE=1 replays the cache and never calls the model. When the model is
unreachable or its answer is not JSON, ask() returns None and the agent falls back to
its rules, saying so in the trace.
"""
import hashlib
import json
import os
import re
from pathlib import Path

from measure import nightshift

CACHE = Path(os.environ.get("NEMOTRON_CACHE", "examples/nemotron_cache.json"))

SYSTEM = (
    "You are the reasoning core of Latent Raw, which turns 8-bit AI video into 32-bit ACES EXR. "
    "You decide only from the evidence rows you are given and cite their evidence_id values. "
    "Expanded range is synthesized, never recovered. Answer with one JSON object and nothing else."
)


def _key(prompt):
    return hashlib.sha1(prompt.encode()).hexdigest()[:16]


def _parse(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def ask(prompt, step, sample_mode):
    """Returns (parsed JSON or None, {"model", "receipt", "note"})."""
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    k = _key(prompt)
    if sample_mode:
        hit = cache.get(k)
        if not hit:
            return None, {"model": None, "receipt": None, "note": "sample mode: no saved answer, used rules"}
        return _parse(hit["text"]), {"model": "replay", "receipt": hit["receipt"], "note": "replayed saved answer"}
    try:
        text, receipt = nightshift.chat([{"role": "system", "content": SYSTEM},
                                         {"role": "user", "content": prompt}], step=step)
    except Exception as e:  # Nightshift down, bad token, timeout
        return None, {"model": None, "receipt": None, "note": f"model unavailable ({type(e).__name__}), used rules"}
    cache[k] = {"step": step, "text": text, "receipt": receipt}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, indent=2, default=str))
    parsed = _parse(text)
    return parsed, {"model": "live", "receipt": receipt,
                    "note": "model answered" if parsed else "model answer was not JSON, used rules"}
