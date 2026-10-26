"""The agent reads evidence only through SQL, so every claim traces to a query and a row.

With EVIDENCE_DATABASE_URL set, rows go to Postgres (the same tables Zetaris reads).
Without it (judges, sample mode), the same rows go to an in-memory SQLite database and
the same SQL runs there. Queries use "?" placeholders; Postgres gets "%s".

EVIDENCE_READ_VIA=zetaris keeps writes going straight to Postgres but sends every read
through Zetaris, where the database is the datasource evidence_pg (ZETARIS_DATASOURCE):
tables become evidence_pg.item_metrics and so on, and the "?" values are written into the
Spark SQL. Sample mode always reads locally.
"""
import json
import os
import re
import sqlite3

from measure import store

from . import zetaris

ORDER = ("shots", "frame_range", "item_metrics", "balance_checks", "vision_checks")


SAFE_TEXT = re.compile(r"^[A-Za-z0-9_@.:\-]*$")  # clip ids, items, stages: nothing to escape
NUMBER = re.compile(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$")


class Evidence:
    def __init__(self, url=None, read_via=None, sample_mode=False):
        url = url if url is not None else os.environ.get("EVIDENCE_DATABASE_URL")
        if url:
            self.kind, self.conn = "postgres", store.connect(url)
            store.create_schema(self.conn)
        else:
            self.kind, self.conn = "sqlite", sqlite3.connect(":memory:")
            self.conn.row_factory = sqlite3.Row
        self.read_via = "local" if sample_mode else (read_via or os.environ.get("EVIDENCE_READ_VIA", "postgres")).lower()
        if self.read_via == "zetaris" and self.kind != "postgres":
            raise RuntimeError("EVIDENCE_READ_VIA=zetaris needs EVIDENCE_DATABASE_URL: "
                               "the agent writes to Postgres and Zetaris reads it back")
        self.datasource = os.environ.get("ZETARIS_DATASOURCE", "evidence_pg")

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
        if self.read_via == "zetaris":
            return self._zetaris(query, params)
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

    # -- Zetaris reads --------------------------------------------------------
    def to_spark(self, query, params):
        """Prefix tables with the datasource and write the values into the SQL."""
        spark = re.sub(r"\b(FROM|JOIN)\s+(\w+)\b", lambda m: f"{m.group(1)} {self.datasource}.{m.group(2)}",
                       " ".join(query.split()), flags=re.I)
        values = iter(params)
        return re.sub(r"\?", lambda m: _literal(next(values)), spark)

    def _zetaris(self, query, params):
        spark = self.to_spark(query, params)
        data, query_id = zetaris.run_sql(spark)
        cols = _select_columns(spark)
        rows = [{c: _cell(v) for c, v in zip(cols, rec)} for rec in data]
        return rows, {"query": " ".join(query.split()), "params": list(params),
                      "via": "zetaris", "sql_sent": spark, "query_id": query_id}


def _literal(v):
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):
        return repr(v)
    v = str(v)
    if not SAFE_TEXT.match(v):
        raise ValueError(f"refusing to send {v!r} to Zetaris: unexpected characters")
    return f"'{v}'"


def _select_columns(sql):
    """Names in SELECT order, since Zetaris may return the column names as null."""
    m = re.search(r"^\s*SELECT\s+(.*?)\s+FROM\s", sql, re.I | re.S)
    if not m or m.group(1).strip() == "*":
        raise ValueError("Zetaris reads need an explicit SELECT column list")
    return [c.strip().split()[-1].split(".")[-1] for c in m.group(1).split(",")]


def _cell(v):
    """Zetaris may send numbers as text; turn them back into numbers."""
    if isinstance(v, str) and NUMBER.match(v):
        return int(v) if re.fullmatch(r"-?\d+", v) else float(v)
    return v
