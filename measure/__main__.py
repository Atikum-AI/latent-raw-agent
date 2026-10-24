"""Command line for measure/.

Measure one frame of a shot and write the rows:
    python -m measure shot --shot shot01 --frame 120 --src frame.png --exr frame.exr \
        --items masks/ --synth synth.png --out examples/shot01_f120.json --db --vision

Compare the same items across two measured shots (for the Verifier):
    python -m measure compare examples/shot01_f120.json examples/shot02_f40.json \
        --stage before --out examples/balance_01_02_before.json --db
"""
import argparse
import json

import numpy as np

from . import metrics, store


def cmd_shot(a):
    src = metrics.load_8bit(a.src)
    shape = src.shape[:2]
    items = metrics.load_item_masks(a.items, shape)
    synth = metrics.load_mask(a.synth, shape) if a.synth else np.zeros(shape, bool)
    exr = metrics.load_exr(a.exr)[1] if a.exr else None
    if exr is not None and exr.shape[:2] != shape:
        raise SystemExit(f"EXR is {exr.shape[:2]}, source frame is {shape}")

    tables = {
        "shots": [{"shot_id": a.shot, "source_model": a.source_model, "source_file": a.source_file}],
        "item_metrics": metrics.item_rows(a.shot, a.frame, src, items, synth, exr, a.aces),
        "exr_headers": [metrics.exr_header_row(a.shot, a.frame, a.exr)] if a.exr else [],
        "frame_range": [metrics.frame_range_row(a.shot, a.frame, src, exr, synth, a.aces)] if exr is not None else [],
        "vision_checks": [],
    }
    if a.vision:
        from . import vision
        tables["vision_checks"] = vision.check_frame(a.shot, a.frame, a.vision_frame or a.src,
                                                     cache_path=a.vision_cache)
    _write(a, tables)


def cmd_compare(a):
    rows_a = json.load(open(a.a))["item_metrics"]
    rows_b = json.load(open(a.b))["item_metrics"]
    tables = {"balance_checks": metrics.balance_rows(rows_a, rows_b, a.stage, use_hdr=not a.sdr)}
    _write(a, tables)


def _write(a, tables):
    if a.out:
        print("saved", store.save_json(a.out, tables))
    if a.db:
        conn = store.connect()
        store.create_schema(conn)
        for t in store.TABLES:  # parents first, so foreign keys hold
            if tables.get(t):
                print(t, store.upsert(conn, t, tables[t]), "rows")
    for t, rows in tables.items():
        for r in rows:
            if t in ("item_metrics", "balance_checks"):
                print(json.dumps(r, default=str))


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m measure")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("shot", help="measure one frame of a shot")
    s.add_argument("--shot", required=True)
    s.add_argument("--frame", type=int, required=True)
    s.add_argument("--src", required=True, help="8-bit frame (PNG/JPEG)")
    s.add_argument("--items", required=True, help="folder of SAM 3.1 masks, one <label>.png per item")
    s.add_argument("--exr", help="32-bit ACES EXR of the same frame")
    s.add_argument("--synth", help="mask of synthesized pixels")
    s.add_argument("--aces", default="AP0", choices=["AP0", "AP1"])
    s.add_argument("--source-model", help="Seedance / Veo / Kling")
    s.add_argument("--source-file")
    s.add_argument("--vision", action="store_true", help="also ask Nemotron about the frame")
    s.add_argument("--vision-frame", help="JPEG to send instead of --src (keep it small)")
    s.add_argument("--vision-cache", default="examples/vision_cache.json")
    s.set_defaults(fn=cmd_shot)

    c = sub.add_parser("compare", help="balance gaps for matched items across two shots")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--stage", required=True, choices=["before", "after"])
    c.add_argument("--sdr", action="store_true", help="compare 8-bit values instead of the EXR")
    c.set_defaults(fn=cmd_compare)

    for q in (s, c):
        q.add_argument("--out", help="save rows as JSON (for sample mode)")
        q.add_argument("--db", action="store_true", help="upsert rows into EVIDENCE_DATABASE_URL")

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
