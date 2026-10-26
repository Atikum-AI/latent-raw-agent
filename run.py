"""Latent Raw: one entry point for judges and for us.

    python run.py --case case01_car_sky           # one eval case
    python run.py --all                           # every case in evals/cases
    python run.py --clips clip03a_canoe_dawn clip03b_jungle_warm --case-id my_match
    python run.py --serve --port 8000             # POST /run (FastAPI + uvicorn)

SAMPLE_MODE=1 (or --sample) replays saved measurements and Nemotron answers, so it
needs no GPU, no Nightshift and no database. Every decision goes to logs/run.jsonl;
score it with:  python -m evals logs/run.jsonl
"""
import argparse
import json
import os
import sys
from pathlib import Path

from agent.core import run_case
from agent.samples import MissingSamples

CASES = Path("evals/cases")


def load_case(case_id):
    return json.loads((CASES / f"{case_id}.json").read_text())


def resolve(body):
    """POST /run body: {"case": "case01_car_sky"} or {"clips": [...], "case_id": "...", "memory": [...]}."""
    if body.get("case"):
        return load_case(body["case"])
    if not body.get("clips"):
        raise ValueError('send {"case": "<case_id>"} or {"clips": [...]}')
    return {"case_id": body.get("case_id", "adhoc"), "clips": body["clips"], "memory": body.get("memory", [])}


def make_app(log_path="logs/run.jsonl"):
    from fastapi import Body, FastAPI, HTTPException

    app = FastAPI(title="Latent Raw agent")

    @app.get("/")
    def health():
        return {"ok": True, "post": "/run", "sample_mode": os.environ.get("SAMPLE_MODE", "0")}

    @app.post("/run")
    def run(body: dict = Body(...)):
        try:
            return run_case(resolve(body), log_path=log_path)
        except (ValueError, FileNotFoundError, KeyError) as e:
            raise HTTPException(400, str(e))

    return app


def main(argv=None):
    p = argparse.ArgumentParser(prog="python run.py")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--case", help="an eval case id, e.g. case01_car_sky")
    g.add_argument("--all", action="store_true", help="run every case in evals/cases")
    g.add_argument("--clips", nargs="+", help="one clip, or two to match")
    g.add_argument("--serve", action="store_true", help="serve POST /run")
    p.add_argument("--case-id", default="adhoc")
    p.add_argument("--sample", action="store_true", help="same as SAMPLE_MODE=1")
    p.add_argument("--log", default="logs/run.jsonl")
    p.add_argument("--fresh", action="store_true", help="clear the log first")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args(argv)

    if a.sample:
        os.environ["SAMPLE_MODE"] = "1"
    if a.fresh and Path(a.log).exists():
        Path(a.log).unlink()
    if a.serve:
        import uvicorn
        print(f"POST http://0.0.0.0:{a.port}/run  (log: {a.log})")
        uvicorn.run(make_app(a.log), host="0.0.0.0", port=a.port)
        return

    if a.all:
        cases = [json.loads(f.read_text()) for f in sorted(CASES.glob("*.json"))]
    elif a.case:
        cases = [load_case(a.case)]
    else:
        cases = [{"case_id": a.case_id, "clips": a.clips, "memory": []}]
    results = []
    for c in cases:
        try:
            out = run_case(c, log_path=a.log)
        except MissingSamples as e:
            sys.exit(f"{c['case_id']}: {e}")
        rep = out["report"]
        print(f"{c['case_id']}: {'PASSED' if rep['passed'] else 'NOT PASSED' if rep['passed'] is False else 'NOTHING TO CHECK'}")
        print(f"    {rep['summary']}")
        results.append(out)
    print(f"trace: {a.log}   score it: python -m evals {a.log}")
    return results


if __name__ == "__main__":
    main()
