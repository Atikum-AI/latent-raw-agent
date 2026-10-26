# agent/ — the Latent Raw reasoning core

`run.py` (repo root) is the one entry point. It runs five steps per case and writes one
JSON line per step to `logs/run.jsonl`, which `python -m evals` scores.

| Step | Reads (SQL) | Asks Nemotron | Checked against |
|---|---|---|---|
| classify | `item_metrics` rows for the clip(s) | which items are in frame, which are clipped | items need a measured row; clipped ≥ 5% of pixels at 250+ stays clipped |
| plan | `frame_range` rows, rulings from memory | pass order, which rulings apply | the brief's hard order (expand on the full frame → ACES → SAM to measure → balance → write); a ruling must cover a measured item |
| run | `item_metrics` ids | — | finds each pass's precomputed output (Kyo's LoRA passes run ahead of time on GPU) |
| verify | `item_metrics` before/after balance, `balance_checks` | — | clipped items got synthesized range; protected items (rulings) not pulled down; two clips within 1/3 stop |
| report | — | a two-line note for the colourist | says "synthesized", never "recovered" |

Each trace line: `case_id, step, decision, cites, sql (query + params), model (receipt), why, run_id, ts`.
The model proposes; the evidence decides. When the model is down or replies with no JSON,
the step falls back to rules and the trace says so.

## Run it

```
python -m agent.demo                                  # generated stand-in clips -> samples_demo/
SAMPLES_DIR=samples_demo python run.py --all --fresh  # all three eval cases
python -m evals logs/run.jsonl                        # score: 100/100 on the stand-ins
python run.py --serve --port 8000                     # FastAPI; then: curl -X POST localhost:8000/run -d '{"case":"case01_car_sky"}'
```

## Reading evidence through Zetaris

```
EVIDENCE_READ_VIA=zetaris   # default: postgres
EVIDENCE_DATABASE_URL=postgresql://latent:latent@localhost:5432/evidence   # writes still go here
ZETARIS_REST_URL=http://localhost:8888
ZETARIS_API_KEY=...
ZETARIS_ORG_ID=1
ZETARIS_DATASOURCE=evidence_pg   # the name the Postgres is registered under in Zetaris
```
Writes stay direct to Postgres. Every read goes to Zetaris's SQL editor API as one Spark SQL
statement, with tables read as `evidence_pg.item_metrics` and so on. Each trace line's `sql`
entry then records `via: zetaris`, the exact SQL sent, and the Zetaris `query_id`.
Sample mode always reads locally, so judges need no Zetaris.

## Where clip files go (agreed with Kyo)

```
samples/<clip_id>/clip.json, source.png, masks/<item>.png, synth.png, expand_hdr.exr, balance.exr
```
`clip.json` sets the key frame and the ACES space (`"aces": "AP1"` for ACEScg). The EXRs come from
the Hugging Face dataset caiotbr/latent-raw-agent-samples: `python scripts/fetch_samples.py`. If a clip has no
`evidence.json` and no EXR, `run.py` prints that command.

## Sample mode for judges

1. Run once live (Nightshift on, real samples): `python run.py --all --fresh`.
   This writes `samples/<clip>/evidence.json` (the measured rows) and
   `examples/nemotron_cache.json` (Nemotron's answers with receipts).
2. Commit those two kinds of file. The EXRs stay in the Hugging Face dataset.
3. Judges run `SAMPLE_MODE=1 python run.py --all` — no GPU, no Nightshift, no database.

Without `EVIDENCE_DATABASE_URL` the same SQL runs on an in-memory SQLite copy of the rows.
