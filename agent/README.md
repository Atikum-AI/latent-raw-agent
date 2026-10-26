# agent/ — the Latent Raw reasoning core

`run.py` (repo root) is the one entry point. It runs five steps per case and writes one
JSON line per step to `logs/run.jsonl`, which `python -m evals` scores.

| Step | Reads (SQL) | Asks Nemotron | Checked against |
|---|---|---|---|
| classify | `item_metrics` rows for the clip(s) | which items are in frame, which are clipped | items need a measured row; clipped ≥ 1% of pixels at 254–255 stays clipped |
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

## Where clip files go (agreed with Kyo)

```
samples/<clip_id>/clip.json, source.png, masks/<item>.png, synth.png, expand_hdr.exr, balance.exr
```
`clip.json` sets the key frame and the ACES space (`"aces": "AP1"` for ACEScg). The EXRs come from
the `sample-data-v1` release: `python scripts/fetch_samples.py`. If a clip has no
`evidence.json` and no EXR, `run.py` prints that command.

## Sample mode for judges

1. Run once live (Nightshift on, real samples): `python run.py --all --fresh`.
   This writes `samples/<clip>/evidence.json` (the measured rows) and
   `examples/nemotron_cache.json` (Nemotron's answers with receipts).
2. Commit those two kinds of file. EXRs and PNGs go to the `sample-data-v1` release.
3. Judges run `SAMPLE_MODE=1 python run.py --all` — no GPU, no Nightshift, no database.

Without `EVIDENCE_DATABASE_URL` the same SQL runs on an in-memory SQLite copy of the rows.
