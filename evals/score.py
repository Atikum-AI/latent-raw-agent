"""Score the agent's decisions against the right plan for each eval case.

A case (evals/cases/*.json) says what the right plan is: which items are in frame,
which are clipped, which passes to run in which order, which colourist rulings apply.
An expected value of null means that check is skipped for the case.

A trace is the agent's JSONL log for one run: one line per decision. The scorer reads
lines shaped like this (FIELDS below maps the names, so it can follow Saad's format):

    {"case_id": "case01_car_sky", "step": "classify",
     "decision": {"items": ["car", "sky"], "clipped_items": ["sky"]},
     "cites": ["item:shot01:f1:sky"]}
    {"case_id": "case01_car_sky", "step": "plan",
     "decision": {"passes": ["expand_hdr", "to_aces_linear", "sam_items", "balance", "write_exr"],
                  "rulings": []},
     "cites": ["item:shot01:f1:sky"]}
    {"case_id": "case01_car_sky", "step": "verify",
     "decision": {"max_lum_gap_stops": 0.2}, "cites": ["balance:shot01:shot02:sky:after"]}

Every check returns a score from 0 to 1 and a reason, so the report says why.
"""
import json
from pathlib import Path

FIELDS = {
    "case": "case_id",
    "step": "step",
    "decision": "decision",
    "cites": "cites",
    "items": "items",
    "clipped": "clipped_items",
    "passes": "passes",
    "rulings": "rulings",
    "gap": "max_lum_gap_stops",
}

# Points per check. Plan and order weigh most: they are the reasoning Track 03 judges.
WEIGHTS = {
    "items": 15,
    "clipped": 15,
    "passes": 20,
    "order": 20,
    "rulings": 15,
    "citations": 15,
    "clip_match": 10,
}

# The brief's hard rule: expand on the full frame before splitting with SAM.
MUST_PRECEDE = [("expand_hdr", "sam_items"), ("expand_hdr", "balance"), ("sam_items", "balance")]


def load_cases(folder):
    return {c["case_id"]: c for c in
            (json.loads(p.read_text()) for p in sorted(Path(folder).glob("*.json")))}


def load_trace(path):
    lines = []
    for raw in Path(path).read_text().splitlines():
        raw = raw.strip()
        if raw:
            lines.append(json.loads(raw))
    return lines


def _decisions(trace, case_id):
    return [l for l in trace
            if l.get(FIELDS["case"]) == case_id and isinstance(l.get(FIELDS["decision"]), dict)]


def _collect(lines, key):
    """Union of a list field across all of a run's decisions, keeping first-seen order."""
    out = []
    for l in lines:
        for v in l[FIELDS["decision"]].get(FIELDS[key]) or []:
            if v not in out:
                out.append(v)
    return out


def _passes(lines):
    for l in lines:
        p = l[FIELDS["decision"]].get(FIELDS["passes"])
        if p:
            return list(p)
    return []


def check_items(case, lines):
    if case["expected"]["items"] is None:
        return None, "not set for this case"
    want = set(case["expected"]["items"])
    got = set(_collect(lines, "items"))
    found = want & got
    return len(found) / len(want) if want else 1.0, f"found {sorted(found)} of {sorted(want)}"


def check_clipped(case, lines):
    if case["expected"]["clipped_items"] is None:
        return None, "not set for this case"
    want = set(case["expected"]["clipped_items"])
    got = set(_collect(lines, "clipped"))
    if want == got:
        return 1.0, f"clipped items match {sorted(want)}"
    return 0.0, f"expected clipped {sorted(want)}, agent said {sorted(got)}"


def check_passes(case, lines):
    want, got = set(case["expected"]["passes"]), set(_passes(lines))
    missing, extra = sorted(want - got), sorted(got - want)
    score = len(want & got) / len(want | got) if want | got else 1.0
    if not missing and not extra:
        return score, "same passes as the right plan"
    return score, f"missing {missing}, extra {extra}"


def check_order(case, lines):
    """Share of pass pairs the agent ran in the right order; a broken hard rule scores 0."""
    plan = _passes(lines)
    pos = {p: i for i, p in enumerate(plan)}
    for a, b in MUST_PRECEDE:
        if a in pos and b in pos and pos[a] > pos[b]:
            return 0.0, f"breaks the hard rule: {a} must run before {b}"
    want = [p for p in case["expected"]["passes"] if p in pos]
    pairs = [(want[i], want[j]) for i in range(len(want)) for j in range(i + 1, len(want))]
    if not pairs:
        return 0.0, "no plan to order"
    good = sum(pos[a] < pos[b] for a, b in pairs)
    return good / len(pairs), f"{good} of {len(pairs)} pass pairs in the right order"


def check_rulings(case, lines):
    want = set(case["expected"]["rulings"])
    got = set(_collect(lines, "rulings"))
    if want == got:
        return 1.0, "applied the right rulings" if want else "no rulings needed, none applied"
    return 0.0, f"expected rulings {sorted(want)}, agent applied {sorted(got)}"


def check_citations(case, lines):
    """No row, no claim: every decision must cite at least one evidence row."""
    cited = sum(bool(l.get(FIELDS["cites"])) for l in lines)
    return cited / len(lines), f"{cited} of {len(lines)} decisions cite evidence"


def check_clip_match(case, lines):
    limit = case["expected"].get(FIELDS["gap"])
    if limit is None:
        return None, "single clip, not scored"
    gaps = [l[FIELDS["decision"]][FIELDS["gap"]] for l in lines
            if l[FIELDS["decision"]].get(FIELDS["gap"]) is not None]
    if not gaps:
        return 0.0, "agent reported no clip-to-clip gap"
    gap = gaps[-1]
    return (1.0 if abs(gap) <= limit else 0.0), f"final gap {gap} stops, limit {limit}"


CHECKS = {
    "items": check_items,
    "clipped": check_clipped,
    "passes": check_passes,
    "order": check_order,
    "rulings": check_rulings,
    "citations": check_citations,
    "clip_match": check_clip_match,
}


def score_case(case, trace):
    """Returns {"case_id", "score" (0-100), "checks": {name: {"score", "reason"}}}."""
    lines = _decisions(trace, case["case_id"])
    if not lines:
        return {"case_id": case["case_id"], "score": 0.0,
                "checks": {}, "note": "no decisions for this case in the trace"}
    checks, earned, possible = {}, 0.0, 0.0
    for name, fn in CHECKS.items():
        s, reason = fn(case, lines)
        checks[name] = {"score": None if s is None else round(s, 3), "reason": reason}
        if s is not None:
            earned += WEIGHTS[name] * s
            possible += WEIGHTS[name]
    return {"case_id": case["case_id"], "score": round(100 * earned / possible, 1), "checks": checks}


def score_all(cases, trace):
    results = [score_case(c, trace) for c in cases.values()]
    mean = sum(r["score"] for r in results) / len(results) if results else 0.0
    return {"mean_score": round(mean, 1), "cases": results}
