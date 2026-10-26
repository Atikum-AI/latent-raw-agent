"""Colourist rulings, held in the Meterless H-MEM engine (agent/hmem.py).

Rulings arrive in the case file ("memory": [...]) or memory/rulings.json, in this shape, and are captured
into H-MEM's long-term tier with provenance; the agent then retrieves them from H-MEM (hybrid ranking),
and after verify feeds the outcome back (helpful / not helpful), all on H-MEM's audit ledger:

    {"ruling_id": "practical_flames_hot", "text": "On this show, practical flames may clip. Never pull them down."}

A ruling covers an item when a word of the ruling names the item ("flames" covers
"lantern_flame"). The model may also apply a ruling; the agent keeps it only if it covers
an item that has a measured row.
"""
import json
import os
import re
from pathlib import Path

STOP = {"on", "this", "the", "a", "an", "may", "never", "them", "it", "show", "down", "pull",
        "clip", "clips", "hot", "keep", "always", "of", "in", "to", "and", "or", "be"}


def _engine():
    from .hmem import HMem
    return HMem(os.environ.get("HMEM_DIR", "memory/hmem"))


def _entities(text):
    return sorted({_stem(w) for w in re.findall(r"[a-z]+", text.lower())} - STOP)


def load_rulings(case=None, path="memory/rulings.json"):
    """Capture the case's rulings into H-MEM, then retrieve the ones relevant to this case."""
    given = list(case["memory"]) if case and case.get("memory") else []
    p = Path(path)
    if not given and p.exists():
        given = json.loads(p.read_text())
    mem = _engine()
    for r in given:
        mem.capture(r["text"], type="preference", layer="long_term",
                    source=f"case:{case.get('case_id')}" if case and case.get("memory") else str(path),
                    entities=_entities(r["text"]), tags=["ruling"], ruling_id=r["ruling_id"], confidence=0.9)
    mem.resolve_conflicts()
    mem.save()
    return [{"ruling_id": r["ruling_id"], "text": r["text"]} for r in given]


def retrieve_for_items(items, clips=()):
    """After classify: retrieve the rulings relevant to the measured items from H-MEM (hybrid ranking)."""
    mem = _engine()
    ents = sorted({_stem(part) for i in items for part in i.lower().split("_")})
    query = "colourist ruling for " + " ".join(items)
    hits = mem.retrieve(query, entities=ents, tags=["ruling"])
    mem.save()
    out = [{"ruling_id": m.ruling_id, "text": m.content, "hmem_id": m.id, "hmem_score": s}
           for s, m in hits if m.ruling_id]
    return out, mem.last_trace


def feedback(rulings, applied, passed):
    """After verify: a ruling that was applied and passed was helpful; applied and failed, not helpful."""
    if not rulings:
        return
    mem = _engine()
    for r in rulings:
        if r.get("hmem_id") in mem.records and r["ruling_id"] in applied and passed is not None:
            mem.feedback(r["hmem_id"], "helpful" if passed else "not_helpful", verify_passed=passed)
    mem.save()


def _stem(w):
    return w[:-1] if w.endswith("s") and len(w) > 3 else w


def covers(ruling, item):
    words = {_stem(w) for w in re.findall(r"[a-z]+", ruling["text"].lower())} - STOP
    words |= set(ruling.get("applies_to", []))
    return any(_stem(t) in words for t in item.lower().split("_"))
