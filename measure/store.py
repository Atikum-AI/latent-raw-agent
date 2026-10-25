"""Write evidence rows to Postgres, and save them as JSON for sample mode.

Zetaris reads the Postgres tables; the agent then reads them through run_sql.
Writes are upserts on evidence_id, so re-measuring a shot replaces its rows.

Env: EVIDENCE_DATABASE_URL, e.g. postgresql://latent:latent@localhost:5432/evidence
"""
import json
import os
from pathlib import Path

TABLES = ("shots", "exr_headers", "frame_range", "item_metrics", "balance_checks", "vision_checks")
JSON_COLUMNS = {"header_json", "receipt", "cites"}


def connect(url=None):
    import psycopg  # only needed when writing to the database
    return psycopg.connect(url or os.environ["EVIDENCE_DATABASE_URL"])


def create_schema(conn):
    sql = (Path(__file__).parent / "schema.sql").read_text()
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def upsert(conn, table, rows):
    """Insert rows into one table; on an existing key, overwrite the row."""
    if table not in TABLES:
        raise ValueError(f"unknown table {table}")
    if not rows:
        return 0
    key = "shot_id" if table == "shots" else "evidence_id"
    cols = list(rows[0].keys())
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != key)
    sql = (f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))}) "
           f"ON CONFLICT ({key}) DO UPDATE SET {updates}")

    def cell(c, v):
        if c in JSON_COLUMNS and v is not None and not isinstance(v, str):
            return json.dumps(v, default=str)
        return v

    with conn.cursor() as cur:
        cur.executemany(sql, [[cell(c, r.get(c)) for c in cols] for r in rows])
    conn.commit()
    return len(rows)


def save_json(path, tables):
    """tables = {"item_metrics": [...], ...}. The same rows, for SAMPLE_MODE replay."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(tables, indent=2, default=str))
    return p
