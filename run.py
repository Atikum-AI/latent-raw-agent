"""Latent Raw: one entry point for judges and for us.

    python run.py --case case01_car_sky           # one eval case
    python run.py --all                           # every case in evals/cases
    python run.py --clips clip03a_canoe_dawn clip03b_jungle_warm --case-id my_match
    python run.py --serve --port 8000             # POST /run

SAMPLE_MODE=1 (or --sample) replays saved measurements and Nemotron answers, so it
needs no GPU, no Nightshift and no database. Every decision goes to logs/run.jsonl;
score it with:  python -m evals logs/run.jsonl
"""
import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from agent.core import run_case

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


class Handler(BaseHTTPRequestHandler):
    log_path = "logs/run.jsonl"

    def do_POST(self):
        if self.path.rstrip("/") != "/run":
            return self._send(404, {"error": "POST /run"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            out = run_case(resolve(body), log_path=self.log_path)
        except (ValueError, FileNotFoundError, KeyError) as e:
            return self._send(400, {"error": str(e)})
        self._send(200, out)

    def do_GET(self):
        self._send(200, {"ok": True, "post": "/run", "sample_mode": os.environ.get("SAMPLE_MODE", "0")})

    def _send(self, code, obj):
        data = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


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
        Handler.log_path = a.log
        print(f"POST http://0.0.0.0:{a.port}/run  (log: {a.log})")
        HTTPServer(("0.0.0.0", a.port), Handler).serve_forever()
        return

    if a.all:
        cases = [json.loads(f.read_text()) for f in sorted(CASES.glob("*.json"))]
    elif a.case:
        cases = [load_case(a.case)]
    else:
        cases = [{"case_id": a.case_id, "clips": a.clips, "memory": []}]
    results = []
    for c in cases:
        out = run_case(c, log_path=a.log)
        rep = out["report"]
        print(f"{c['case_id']}: {'PASSED' if rep['passed'] else 'NOT PASSED' if rep['passed'] is False else 'NOTHING TO CHECK'}")
        print(f"    {rep['summary']}")
        results.append(out)
    print(f"trace: {a.log}   score it: python -m evals {a.log}")
    return results


if __name__ == "__main__":
    main()
