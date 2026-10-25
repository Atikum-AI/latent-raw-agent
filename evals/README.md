# evals/

Scores the agent's decisions against clips whose right plan we already know. This
is how we show the judges the agent makes reliable decisions, not guesses.

| File | What it does |
| --- | --- |
| `cases/*.json` | One file per eval clip: the items in frame, which are clipped, the passes in order, and the colourist rulings that apply. |
| `score.py` | Compares one agent run with each case and gives 0 to 100 points, with a reason for every check. |
| `__main__.py` | Command line: `python -m evals <trace.jsonl>`. |
| `sample_trace.jsonl` | A made-up agent run for trying the scorer. Case 2 is wrong on purpose. |

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

## To do with Kyo

The three cases use placeholder clip names (`TODO_kyo_clip_01`...). Kyo replaces
them with the real demo clips and corrects the expected plan for each.
