# measure/

Per-item measurements and Nemotron vision checks for the Latent Raw agent. Every
number becomes a row with an `evidence_id`, stored in Postgres, read by Zetaris,
and cited by the agent through `run_sql`. Expanded range is "synthesized", never
"recovered".

| File | What it does |
| --- | --- |
| `metrics.py` | Measures each SAM 3.1 item: clipped %, mean and peak luminance, saturation, colour balance (R/G, B/G), synthesized %, in the 8-bit source and the EXR. `balance_rows()` compares the same items across two clips for the Verifier. |
| `vision.py` | Asks Nemotron Super VL fixed questions about a key frame (contents, highlights, seams). Saves answers to a cache that `SAMPLE_MODE=1` replays. |
| `nightshift.py` | Nightshift client. Text through `/chat/completions` (app `latent-raw`), images through `/jobs` (app `latent-raw-vision`). Logs each call with its receipt to `logs/nightshift.jsonl`. |
| `store.py` | Upserts rows into Postgres and saves the same rows as JSON for sample mode. |
| `schema.sql` | Tables: `shots`, `exr_headers`, `frame_range`, `item_metrics`, `balance_checks`, `vision_checks`, plus the `shot_evidence` view. |
| `__main__.py` | Command line: `python -m measure shot ...` and `python -m measure compare ...`. |

## Measure one frame

SAM masks go in a folder, one white-on-black PNG per item, named by label
(`car.png`, `sky.png`), the same size as the frame.

```
python -m measure shot --shot shot01 --frame 120 \
  --src frames/shot01_f120.png --exr exr/shot01_f120.exr \
  --items masks/shot01_f120/ --synth masks/shot01_f120_synth.png \
  --source-model Veo --out examples/shot01_f120.json --db
```

Add `--vision --vision-frame frames/shot01_f120.jpg` to ask Nemotron about the frame too.

## Compare two clips

```
python -m measure compare examples/shot01_f120.json examples/shot02_f40.json \
  --stage before --out examples/balance_01_02_before.json --db
```

`lum_gap_stops` is how many stops brighter clip B's item is than clip A's. Run it
again with `--stage after` once the balance pass has run.

## Env

```
EVIDENCE_DATABASE_URL=postgresql://latent:latent@localhost:5432/evidence
METERLESS_BASE_URL=http://127.0.0.1:8088/v1
METERLESS_RUNTIME_TOKEN=
SAMPLE_MODE=0
```

## Tests

```
pip install -r measure/requirements.txt
pytest tests/test_measure.py   # set EVIDENCE_DATABASE_URL to include the database test
```
