"""The agent reads evidence only through SQL, so every claim traces to a query and a row.

With EVIDENCE_DATABASE_URL set, rows go to Postgres (the same tables Zetaris reads).
Without it (judges, sample mode), the same rows go to an in-memory SQLite database and
the same SQL runs there. Queries use "?" placeholders; Postgres gets "%s".
"""
import json
import os
import sqlite3

from measure import store

ORDER = ("shots", "frame_range", "item_metrics", "balance_checks", "vision_checks")


class Evidence:
    def __init__(self, url=None):
        url = url if url is not None else os.environ.get("EVIDENCE_DATABASE_URL")
        if url:
            self.kind, self.conn = "postgres", store.connect(url)
            store.create_schema(self.conn)
        else:
            self.kind, self.conn = "sqlite", sqlite3.connect(":memory:")
            self.conn.row_factory = sqlite3.Row

    def load(self, tables):
        if self.kind == "postgres" and tables.get("shots"):
            self._clear([s["shot_id"] for s in tables["shots"]])
        for t in ORDER:
            rows = tables.get(t) or []
            if not rows:
                continue
            if self.kind == "postgres":
                store.upsert(self.conn, t, rows)
            else:
                self._sqlite_insert(t, rows)

    def _clear(self, shot_ids):
        """Drop a clip's old measurements, so rows from an earlier key frame or an
        earlier run can't be read as evidence for this one."""
        with self.conn.cursor() as cur:
            for table, where in (("balance_checks", "shot_a = ANY(%s) OR shot_b = ANY(%s)"),
                                 ("item_metrics", "shot_id = ANY(%s)"),
                                 ("frame_range", "shot_id = ANY(%s)")):
                cur.execute(f"DELETE FROM {table} WHERE {where}", [shot_ids] * where.count("%s"))
        self.conn.commit()

    def _sqlite_insert(self, table, rows):
        cols = sorted({c for r in rows for c in r})
        self.conn.execute(f"CREATE TABLE IF NOT EXISTS {table} ({', '.join(cols)})")
        have = {r[1] for r in self.conn.execute(f"PRAGMA table_info({table})")}
        for c in cols:
            if c not in have:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {c}")
        cell = lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v
        self.conn.executemany(
            f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
            [[cell(r.get(c)) for c in cols] for r in rows])

    def sql(self, query, params=()):
        """Run one query. Returns (rows as dicts, {"query", "params"} for the trace)."""
        if self.kind == "postgres":
            with self.conn.cursor() as cur:
                cur.execute(query.replace("?", "%s"), params)
                names = [d.name for d in cur.description]
                rows = [dict(zip(names, r)) for r in cur.fetchall()]
        else:
            rows = [dict(r) for r in self.conn.execute(query, params)]
        rows = [{k: float(v) if hasattr(v, "as_integer_ratio") and not isinstance(v, (int, float)) else v
                 for k, v in r.items()} for r in rows]  # Postgres numeric -> float
        return rows, {"query": " ".join(query.split()), "params": list(params)}
