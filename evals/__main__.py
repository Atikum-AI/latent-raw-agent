"""Command line for evals/.

Score an agent run against every case:
    python -m evals logs/run.jsonl
    python -m evals logs/run.jsonl --cases evals/cases --out evals/report.json
"""
import argparse
import json

from .score import load_cases, load_trace, score_all


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m evals")
    p.add_argument("trace", help="the agent's JSONL decision log")
    p.add_argument("--cases", default="evals/cases")
    p.add_argument("--out", help="save the full report as JSON")
    a = p.parse_args(argv)

    report = score_all(load_cases(a.cases), load_trace(a.trace))
    for r in report["cases"]:
        print(f"{r['case_id']:<28} {r['score']:>5}/100")
        for name, c in r["checks"].items():
            mark = "-" if c["score"] is None else f"{c['score']:.2f}"
            print(f"    {name:<11} {mark:>5}  {c['reason']}")
        if r.get("note"):
            print(f"    {r['note']}")
    print(f"{'mean':<28} {report['mean_score']:>5}/100")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=2)
        print("saved", a.out)
    return report


if __name__ == "__main__":
    main()
