"""Latent Raw: one entry point for judges and for us.

    python run.py --case case01_car_sky           # one eval case
    python run.py --all                           # every case in evals/cases
    python run.py --clips clip03a_canoe_dawn clip03b_jungle_warm --case-id my_match
    python run.py --serve --port 8000             # POST /run on 127.0.0.1 (FastAPI + uvicorn)
    RUN_TOKEN=... python run.py --serve --host 0.0.0.0   # on the network: Authorization: Bearer <RUN_TOKEN>

SAMPLE_MODE=1 (or --sample) replays saved measurements and Nemotron answers, so it
needs no GPU, no Nightshift and no database. Every decision goes to logs/run.jsonl;
score it with:  python -m evals logs/run.jsonl
"""
import argparse
import hmac
import json
import os
import sys
from pathlib import Path

from agent.core import run_case
from agent.samples import MissingSamples, check_name

CASES = Path("evals/cases")
MAX_RULINGS = 20          # rulings one POST /run may add to memory
MAX_RULING_CHARS = 500
LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def load_case(case_id):
    return json.loads((CASES / f"{check_name(case_id, 'case id')}.json").read_text())


def resolve(body):
    """POST /run body: {"case": "case01_car_sky"} or {"clips": [...], "case_id": "...", "memory": [...]}."""
    if body.get("case"):
        return load_case(body["case"])
    if not body.get("clips"):
        raise ValueError('send {"case": "<case_id>"} or {"clips": [...]}')
    clips = [check_name(c) for c in body["clips"]]
    memory = body.get("memory", [])
    if not isinstance(memory, list) or len(memory) > MAX_RULINGS:
        raise ValueError(f"memory takes at most {MAX_RULINGS} rulings")
    for r in memory:
        if not isinstance(r, dict) or not isinstance(r.get("text"), str) or len(r["text"]) > MAX_RULING_CHARS:
            raise ValueError(f"each ruling needs a text of at most {MAX_RULING_CHARS} characters")
        check_name(r.get("ruling_id"), "ruling_id")
    return {"case_id": check_name(body.get("case_id", "adhoc"), "case_id"), "clips": clips, "memory": memory}


def make_app(log_path="logs/run.jsonl"):
    from fastapi import Body, FastAPI, Header, HTTPException

    app = FastAPI(title="Latent Raw agent")

    @app.get("/")
    def health():
        return {"ok": True, "post": "/run", "sample_mode": os.environ.get("SAMPLE_MODE", "0")}

    @app.post("/run")
    def run(body: dict = Body(...), authorization: str = Header("")):
        token = os.environ.get("RUN_TOKEN", "")   # set it to require  Authorization: Bearer <RUN_TOKEN>
        if token and not hmac.compare_digest(authorization.encode(), f"Bearer {token}".encode()):
            raise HTTPException(401, "send Authorization: Bearer <RUN_TOKEN>")
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
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to serve the network (needs RUN_TOKEN)")
    a = p.parse_args(argv)

    if a.sample:
        os.environ["SAMPLE_MODE"] = "1"
    if a.fresh and Path(a.log).exists():
        Path(a.log).unlink()
    if a.serve:
        if a.host not in LOOPBACK and not os.environ.get("RUN_TOKEN"):
            sys.exit(f"--host {a.host} serves the network: set RUN_TOKEN first")
        import uvicorn
        print(f"POST http://{a.host}:{a.port}/run  (log: {a.log})")
        uvicorn.run(make_app(a.log), host=a.host, port=a.port)
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
