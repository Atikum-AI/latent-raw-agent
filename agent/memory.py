"""Colourist rulings. Kyo's H-MEM engine holds them; until it is wired in, rulings come
from the case file ("memory": [...]) or memory/rulings.json, in the same shape:

    {"ruling_id": "practical_flames_hot", "text": "On this show, practical flames may clip. Never pull them down."}

A ruling covers an item when a word of the ruling names the item ("flames" covers
"lantern_flame"). The model may also apply a ruling; the agent keeps it only if it covers
an item that has a measured row.
"""
import json
import re
from pathlib import Path

STOP = {"on", "this", "the", "a", "an", "may", "never", "them", "it", "show", "down", "pull",
        "clip", "clips", "hot", "keep", "always", "of", "in", "to", "and", "or", "be"}


def load_rulings(case=None, path="memory/rulings.json"):
    if case and case.get("memory"):
        return list(case["memory"])
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else []


def _stem(w):
    return w[:-1] if w.endswith("s") and len(w) > 3 else w


def covers(ruling, item):
    words = {_stem(w) for w in re.findall(r"[a-z]+", ruling["text"].lower())} - STOP
    words |= set(ruling.get("applies_to", []))
    return any(_stem(t) in words for t in item.lower().split("_"))
