"""The orchestrator: classify -> plan -> run -> verify -> report.

Each step reads evidence through SQL, may ask Nemotron, checks the answer against the
evidence and the brief's hard rules, and writes one trace line:

    {"case_id", "step", "decision", "cites", "sql", "model", "why", "run_id", "ts"}

The decision fields (items, clipped_items, passes, rulings, max_lum_gap_stops) are the
ones evals/score.py reads. "No row, no claim": the model may only name items that have
a measured row, and a step that cannot cite a row says so.
"""
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from measure import metrics

from . import memory, reason, samples
from .evidence import Evidence

PIPELINE = ["expand_hdr", "to_aces_linear", "sam_items", "balance", "write_exr"]
# The brief's hard rules: expand on the full frame, then convert, then split with SAM
# to measure (never to regenerate), then balance, then write.
MUST_PRECEDE = [("expand_hdr", "to_aces_linear"), ("expand_hdr", "sam_items"), ("expand_hdr", "balance"),
                ("sam_items", "balance"), ("balance", "write_exr"), ("to_aces_linear", "write_exr")]
CLIPPED_PCT = 1.0      # an item with >= 1% of its pixels at 254-255 is clipped
MAX_GAP_STOPS = 0.33   # two clips cut together when matched items sit within 1/3 stop
PROTECT_KEEP = 0.95    # a protected item keeps at least 95% of its peak through balance


def _in(n):
    return ", ".join("?" * n)


def _plan_ok(passes):
    if sorted(passes) != sorted(PIPELINE):
        return False
    pos = {p: i for i, p in enumerate(passes)}
    return all(pos[a] < pos[b] for a, b in MUST_PRECEDE)


class Run:
    def __init__(self, case_id, clips, rulings=(), sample_mode=False, log_path="logs/run.jsonl",
                 evidence=None):
        self.case_id, self.clips, self.rulings = case_id, list(clips), list(rulings)
        self.sample_mode, self.log_path = sample_mode, Path(log_path)
        self.ev = evidence or Evidence()
        self.run_id = uuid.uuid4().hex[:8]
        self.lines = []

    # -- trace ---------------------------------------------------------------
    def log(self, step, decision, cites, sql, model, why):
        line = {"ts": datetime.now(timezone.utc).isoformat(), "run_id": self.run_id,
                "case_id": self.case_id, "step": step, "decision": decision, "cites": cites,
                "sql": sql, "model": model, "why": why}
        self.lines.append(line)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(line, default=str) + "\n")
        return line

    # -- steps ---------------------------------------------------------------
    def load(self):
        for c in self.clips:
            self.ev.load(samples.load_evidence(c, self.sample_mode))

    def classify(self):
        rows, q = self.ev.sql(
            f"SELECT evidence_id, shot_id, item, area_pct, clipped_pct, synthesized_pct "
            f"FROM item_metrics WHERE shot_id IN ({_in(len(self.clips))}) ORDER BY shot_id, item",
            self.clips)
        measured = list(dict.fromkeys(r["item"] for r in rows))
        clip_pct = {}
        for r in rows:
            clip_pct[r["item"]] = max(clip_pct.get(r["item"], 0.0), float(r["clipped_pct"] or 0))
        rule_clipped = [i for i in measured if clip_pct[i] >= CLIPPED_PCT]

        prompt = (
            "Step: classify. Evidence rows from item_metrics (one per SAM item; clipped_pct is the "
            "share of the item's 8-bit pixels at 254-255):\n" + json.dumps(rows) +
            f"\nWhich items are in frame, and which are clipped (treat >= {CLIPPED_PCT}% as clipped "
            "unless you see a reason in the rows)? Reply as JSON: "
            '{"items": [...], "clipped_items": [...], "why": "one sentence citing evidence_id values"}'
        )
        ans, model = reason.ask(prompt, "classify", self.sample_mode)
        items, clipped, why = measured, rule_clipped, "rules: every SAM item with a row; clipped by threshold"
        if ans:
            unbacked = [i for i in ans.get("items", []) + ans.get("clipped_items", []) if i not in clip_pct]
            # The model may call an item clipped if its row shows any clipping; the
            # threshold items stay clipped whatever it says.
            extra = [i for i in ans.get("clipped_items", []) if i in clip_pct and clip_pct[i] > 0]
            clipped = list(dict.fromkeys(rule_clipped + extra))
            why = ans.get("why", "")
            if unbacked:
                why += f" [dropped {unbacked}: no measured row]"
        cites = [r["evidence_id"] for r in rows if r["item"] in clipped] or [r["evidence_id"] for r in rows]
        self.items, self.clipped = items, clipped
        return self.log("classify", {"items": items, "clipped_items": clipped}, cites, [q], model, why)

    def plan(self):
        rows, q = self.ev.sql(
            f"SELECT evidence_id, shot_id, src_clipped_pct, hdr_range_stops, synthesized_pct "
            f"FROM frame_range WHERE shot_id IN ({_in(len(self.clips))})", self.clips) \
            if self._has("frame_range") else ([], None)
        valid = {r["ruling_id"]: r for r in self.rulings if any(memory.covers(r, i) for i in self.items)}

        prompt = (
            "Step: plan. Items: " + json.dumps(self.items) + ". Clipped: " + json.dumps(self.clipped) +
            ". Frame range rows: " + json.dumps(rows) + ". Colourist rulings in memory: " +
            json.dumps(self.rulings) + ". Passes available: " + json.dumps(PIPELINE) +
            ". Hard rules: expand_hdr runs on the full frame before sam_items; SAM cut-outs are "
            "measured, never regenerated; balance runs after sam_items; write_exr is last. "
            'Reply as JSON: {"passes": [...in order], "rulings": [ruling_id...], '
            '"protect": [items the balance pass must not pull down], "why": "one sentence"}'
        )
        ans, model = reason.ask(prompt, "plan", self.sample_mode)
        passes, rulings = list(PIPELINE), list(valid)
        why = "rules: the brief's pass order; rulings whose words name a measured item"
        if ans:
            why = ans.get("why", "")
            if _plan_ok(ans.get("passes", [])):
                passes = list(ans["passes"])
            else:
                why += f" [model order {ans.get('passes')} breaks the hard rules; used {PIPELINE}]"
            unknown = [r for r in ans.get("rulings", []) if r not in valid]
            if unknown:
                why += f" [ignored rulings {unknown}: no measured item they cover]"
        protect = [i for i in self.items if any(memory.covers(valid[r], i) for r in rulings)]
        self.passes, self.rulings_applied, self.protect = passes, rulings, protect
        cites = [r["evidence_id"] for r in rows] + [f"hmem:{r}" for r in rulings]
        if not cites:  # no range rows (no EXR yet): cite the clipped items that drove the plan
            cites = self.lines[0]["cites"]
        return self.log("plan", {"passes": passes, "rulings": rulings, "protect": protect},
                        cites, [q] if q else [], model, why)

    def run(self):
        """Kyo's LoRA passes run on his GPU ahead of time; this step finds each output."""
        ran, cites = [], []
        for p in self.passes:
            for c in self.clips:
                out = samples.pass_output(c, p)
                status = "precomputed" if out else (
                    "in release (measured rows saved)" if self.sample_mode else "missing")
                ran.append({"pass": p, "clip": c, "output": str(out) if out else None, "status": status})
        rows, q = self.ev.sql(
            f"SELECT evidence_id FROM item_metrics WHERE shot_id IN ({_in(len(self.clips))}) ORDER BY evidence_id",
            self.clips)
        cites = [r["evidence_id"] for r in rows]
        missing = [r for r in ran if r["status"] == "missing"]
        why = f"{len(ran) - len(missing)} of {len(ran)} pass outputs found"
        if missing:
            why += "; missing: " + ", ".join(f"{m['pass']}@{m['clip']}" for m in missing)
        return self.log("run", {"ran": ran}, cites, [q], {"model": None, "receipt": None, "note": "no model call"}, why)

    def verify(self):
        shots = self.clips + [c + samples.AFTER for c in self.clips]
        rows, q = self.ev.sql(
            f"SELECT evidence_id, shot_id, item, peak_lum, hdr_peak_lum, hdr_mean_lum, hdr_r_over_g, "
            f"hdr_b_over_g, frame FROM item_metrics WHERE shot_id IN ({_in(len(shots))})", shots)
        by = {(r["shot_id"], r["item"]): r for r in rows}
        checks, cites, sql = [], [], [q]

        for c in self.clips:
            for item in self.clipped:
                r = by.get((c, item))
                if not r or r["hdr_peak_lum"] is None:
                    continue
                ok = r["hdr_peak_lum"] > 1.05 * r["peak_lum"]
                checks.append({"check": "range synthesized", "clip": c, "item": item, "passed": ok,
                               "source_peak": r["peak_lum"], "hdr_peak": r["hdr_peak_lum"]})
                cites.append(r["evidence_id"])
            for item in self.protect:
                a, b = by.get((c, item)), by.get((c + samples.AFTER, item))
                if not a or not b or a["hdr_peak_lum"] is None:
                    continue
                ok = b["hdr_peak_lum"] >= PROTECT_KEEP * a["hdr_peak_lum"]
                checks.append({"check": "ruling kept: not pulled down", "clip": c, "item": item, "passed": ok,
                               "peak_before": a["hdr_peak_lum"], "peak_after": b["hdr_peak_lum"]})
                cites += [a["evidence_id"], b["evidence_id"]]

        decision = {}
        if len(self.clips) == 2:
            before, before_cites, bq = self._clip_gap(rows, "before")
            gap, gap_cites, gq = self._clip_gap(rows, "after")
            sql += bq + gq
            if gap is None:  # no balanced EXR yet: judge the clips as they are
                gap, gap_cites = before, before_cites
            if gap is not None:
                ok = abs(gap) <= MAX_GAP_STOPS
                decision["max_lum_gap_stops"] = gap
                check = {"check": "clips match", "passed": ok, "max_lum_gap_stops": gap,
                         "gap_before_balance": before, "limit": MAX_GAP_STOPS}
                if not ok:  # what the next balance pass should do
                    check["fix"] = f"scale {self.clips[1]} by {2 ** -gap:.3f} ({-gap:+.2f} stops) and re-run balance"
                checks.append(check)
                cites += gap_cites
        passed = all(c["passed"] for c in checks) if checks else None
        decision.update({"checks": checks, "passed": passed})
        why = (f"{sum(c['passed'] for c in checks)} of {len(checks)} checks passed" if checks
               else "nothing to verify: no EXR measurements")
        if not cites:
            cites = [r["evidence_id"] for r in rows]
        self.verdict = decision
        return self.log("verify", decision, cites, sql, {"model": None, "receipt": None, "note": "numbers only"}, why)

    def _clip_gap(self, rows, stage):
        """Largest luminance gap (stops) between matched items of the two clips, before or after balance."""
        a, b = self.clips
        sa, sb = (a, b) if stage == "before" else (a + samples.AFTER, b + samples.AFTER)
        side = lambda s: [dict(r) for r in rows if r["shot_id"] == s and r["hdr_mean_lum"] is not None]
        if not side(sa) or not side(sb):
            return None, [], []
        full = {r["evidence_id"]: r for r in self._item_rows([sa, sb])}
        bal = metrics.balance_rows([full[r["evidence_id"]] for r in side(sa)],
                                   [full[r["evidence_id"]] for r in side(sb)], stage)
        if not bal:
            return None, [], []
        self.ev.load({"balance_checks": bal})
        got, q = self.ev.sql("SELECT evidence_id, item, lum_gap_stops FROM balance_checks "
                             "WHERE shot_a = ? AND shot_b = ? AND stage = ?", (sa, sb, stage))
        worst = max(got, key=lambda r: abs(float(r["lum_gap_stops"])))
        return round(float(worst["lum_gap_stops"]), 3), [r["evidence_id"] for r in got], [q]

    def _item_rows(self, shots):
        rows, _ = self.ev.sql(f"SELECT * FROM item_metrics WHERE shot_id IN ({_in(len(shots))})", shots)
        return rows

    def report(self):
        deliver = {}
        for c in self.clips:
            out = samples.pass_output(c, "write_exr") or samples.pass_output(c, "expand_hdr")
            deliver[c] = str(out) if out else f"{c}: final EXR in the sample-data-v1 release"
        facts = {"items": self.items, "clipped": self.clipped, "passes": self.passes,
                 "rulings": self.rulings_applied, "verify": self.verdict}
        prompt = ("Step: report. Write two sentences for the colourist: what was done and whether it "
                  "passed. Say 'synthesized', never 'recovered'. Facts: " + json.dumps(facts) +
                  '. Reply as JSON: {"summary": "..."}')
        ans, model = reason.ask(prompt, "report", self.sample_mode)
        summary = (ans or {}).get("summary") or (
            f"Expanded the full frame{' (synthesized range in ' + ', '.join(self.clipped) + ')' if self.clipped else ''}, measured "
            f"{len(self.items)} items, balanced and wrote ACES EXR"
            + (f", keeping rulings {self.rulings_applied}" if self.rulings_applied else "")
            + f". Verify {'passed' if self.verdict['passed'] else 'did not pass' if self.verdict['passed'] is False else 'had nothing to check'}.")
        if "recovered" in summary.lower():
            summary = summary.replace("recovered", "synthesized").replace("Recovered", "Synthesized")
        cites = self.lines[-1]["cites"]
        return self.log("report", {"deliverables": deliver, "passed": self.verdict["passed"], "summary": summary},
                        cites, [], model, "final summary")

    def _has(self, table):
        try:
            self.ev.sql(f"SELECT 1 FROM {table} LIMIT 1")
            return True
        except Exception:
            if self.ev.kind == "postgres":
                self.ev.conn.rollback()
            return False

    def go(self):
        self.load()
        self.classify(); self.plan(); self.run(); self.verify()
        return self.report()


def run_case(case, sample_mode=None, log_path="logs/run.jsonl"):
    """case = an evals case dict, or {"case_id", "clips", "memory"} from POST /run."""
    if sample_mode is None:
        sample_mode = os.environ.get("SAMPLE_MODE", "0") == "1"
    r = Run(case["case_id"], case["clips"], memory.load_rulings(case), sample_mode, log_path)
    last = r.go()
    return {"case_id": r.case_id, "run_id": r.run_id, "report": last["decision"], "trace": r.lines}
