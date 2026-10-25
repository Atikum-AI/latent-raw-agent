"""Tests for evals/. Run from the repo root: pytest tests/test_evals.py"""
import json

from evals.score import load_cases, load_trace, score_case, score_all
from evals.__main__ import main

CASES = load_cases("evals/cases")


def line(case_id, step, decision, cites=("item:x",)):
    return {"case_id": case_id, "step": step, "decision": decision, "cites": list(cites)}


def right_run(case):
    e = case["expected"]
    out = [line(case["case_id"], "classify", {"items": e["items"] or [], "clipped_items": e["clipped_items"] or []}),
           line(case["case_id"], "plan", {"passes": e["passes"], "rulings": e["rulings"]})]
    if "max_lum_gap_stops" in e:
        out.append(line(case["case_id"], "verify", {"max_lum_gap_stops": 0.1}))
    return out


def test_three_cases_load():
    assert len(CASES) >= 3
    for c in CASES.values():
        assert {"items", "clipped_items", "passes", "rulings"} <= set(c["expected"])
        assert not any("TODO" in clip for clip in c["clips"])


def test_right_plan_scores_100():
    for c in CASES.values():
        assert score_case(c, right_run(c))["score"] == 100.0


def test_sam_before_expand_breaks_the_hard_rule():
    c = CASES["case01_car_sky"]
    run = right_run(c)
    run[1]["decision"]["passes"] = ["sam_items", "expand_hdr", "to_aces_linear", "balance", "write_exr"]
    r = score_case(c, run)
    assert r["checks"]["order"]["score"] == 0.0 and "hard rule" in r["checks"]["order"]["reason"]


def test_missing_ruling_and_citation_lose_points():
    c = CASES["case02_lantern_flame"]
    run = right_run(c)
    run[1]["decision"]["rulings"] = []
    run[1]["cites"] = []
    r = score_case(c, run)
    assert r["checks"]["rulings"]["score"] == 0.0
    assert r["checks"]["citations"]["score"] == 0.5
    assert r["score"] < 100


def test_clip_gap_over_limit_fails():
    c = CASES["case03_dawn_jungle_match"]
    run = right_run(c)
    run[-1]["decision"]["max_lum_gap_stops"] = 0.8
    assert score_case(c, run)["checks"]["clip_match"]["score"] == 0.0


def test_case_missing_from_trace_scores_zero():
    r = score_all(CASES, [])
    assert r["mean_score"] == 0.0


def test_cli_on_sample_trace(tmp_path, capsys):
    out = tmp_path / "report.json"
    report = main(["evals/sample_trace.jsonl", "--out", str(out)])
    assert json.loads(out.read_text())["mean_score"] == report["mean_score"]
    scores = {r["case_id"]: r["score"] for r in report["cases"]}
    assert scores["case01_car_sky"] == 100.0 and scores["case02_lantern_flame"] < 100


def test_null_expectation_skips_the_check():
    c = CASES["case03_dawn_jungle_match"]
    r = score_case(c, right_run(c))
    assert r["checks"]["items"]["score"] is None and r["score"] == 100.0
