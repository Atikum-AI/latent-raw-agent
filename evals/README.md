# evals/

Scores the agent's decisions against clips whose right plan we already know. This
is how we show the judges the agent makes reliable decisions, not guesses.

| File | What it does |
| --- | --- |
| `cases/*.json` | One file per eval clip: the items in frame, which are clipped, the passes in order, and the colourist rulings that apply. |
| `score.py` | Compares one agent run with each case and gives 0 to 100 points, with a reason for every check. |
| `__main__.py` | Command line: `python -m evals <trace.jsonl>`. |
| `sample_trace.jsonl` | A made-up agent run for trying the scorer. Case 2 is wrong on purpose (SAM before expand, ruling missed, one uncited decision). |

## What gets scored

| Check | Points | Passes when |
| --- | --- | --- |
| items | 15 | The agent named the items in frame (car, sky...) |
| clipped | 15 | It found exactly the clipped items |
| passes | 20 | It picked the same passes as the right plan |
| order | 20 | Passes run in the right order. Expanding after SAM, or balancing before SAM, scores 0 (brief section 03) |
| rulings | 15 | It applied the colourist rulings from H-MEM, and only those |
| citations | 15 | Every decision cites at least one evidence row ("no row, no claim") |
| clip_match | 10 | For multi-clip cases: the final brightness gap between clips is under the limit |

## Run it

```
python -m evals evals/sample_trace.jsonl
python -m evals logs/run.jsonl --out evals/report.json
```

## What the scorer reads from the agent's trace

One JSON line per decision, with `case_id`, `step`, `decision` and `cites`. The
decision fields it looks for are `items`, `clipped_items`, `passes`, `rulings` and
`max_lum_gap_stops`. If Saad's trace uses other names, change `FIELDS` at the top
of `score.py`; nothing else needs to change.

## The cases

| Case | Clips | What it tests |
| --- | --- | --- |
| case01_car_sky | clip01_car_street_sun | Sky clipped under a low sun |
| case02_lantern_flame | clip02_lantern_table | Ruling `practical_flames_hot`: leave the flame hot and cite the ruling |
| case03_dawn_jungle_match | clip03a_canoe_dawn, clip03b_jungle_warm | Match skin and exposure across a cool and a warm clip (720p/24 vs 1080p/30) |

All right plans come from Kyo. A `null` expectation skips that check; case03 has no
item list yet, and its 1/3 stop limit is our own choice.
