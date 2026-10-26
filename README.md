# latent-raw-agent
Orchestrator agent for AI footage: classifies 8-bit clips, plans and runs the passes, balances them, and delivers 32-bit ACES EXR with a log of every decision. Open Agent Hackathon 2026, Track 03.
## Quick start (sample mode)

Sample mode replays saved measurements and Nemotron answers, so it needs no GPU, no
Nightshift and no database.

**With Docker**

```
docker build -t latent-raw .
docker run --rm latent-raw
```

**Without Docker** (Python 3.12)

```
pip install -r requirements.txt
SAMPLE_MODE=1 python run.py --all
python -m evals logs/run.jsonl
```

`run.py --all` runs the three eval cases (classify → plan → run → verify → report) and
writes one JSON line per decision to `logs/run.jsonl`: the decision, the evidence rows it
cites, the SQL that found them, and the Nemotron receipt. `python -m evals` scores those
decisions against the right plan for each case.

To serve it: `python run.py --serve --port 8000`, then
`curl -X POST localhost:8000/run -H 'Content-Type: application/json' -d '{"case": "case01_car_sky"}'`.

The full-size clips are in the Hugging Face dataset caiotbr/latent-raw-agent-samples. Sample mode does not need them;
to measure from them yourself, run `python scripts/fetch_samples.py`, then `python run.py --all`
with `SAMPLE_MODE=0`.
