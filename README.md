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

## Memory: Meterless H-MEM

The colourist's rulings ("on this show, practical flames may clip; never pull them down") live in a
[Meterless H-MEM](https://github.com/meterless/H-MEM) engine, implemented in `agent/hmem.py` from the spec's `AGENTS.md`:

- **Canonical records** with provenance: source, confidence, domain, entities, `supersedes`, `relatedTo`, `derivedFrom`.
- **Tiered storage:** short-term (this run), working and long-term (`memory/hmem/*.json`), forgetting curve on non-long-term.
- **Hybrid retrieval** after classify: the measured items (`lantern_flame`) are the query entities, so a ruling about
  "flames" is found for a lantern shot; the plan cites it as `hmem:<ruling_id>`.
- **Trust ledger:** every create, read, feedback and conflict is appended to `memory/hmem/ledger.jsonl`.
- **Feedback:** after verify, an applied ruling that passed gains confidence; one that failed loses it.
- **Conflicts:** opposing rulings on the same items are detected and resolved by recency and confidence (audited).

`python scripts/hmem_demo.py` shows a colourist reversing a ruling, H-MEM detecting the conflict, and the agent
following the newer ruling. Dreaming and sleep cycles (opt-in in the spec) are not implemented.
