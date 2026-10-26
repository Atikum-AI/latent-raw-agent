"""Read evidence through Zetaris (the brief: the agent reads its evidence via Zetaris).

Contract (Kyo, tested live 22 Oct):
    POST {ZETARIS_REST_URL}/api/v1.0/sql-editor/sqls/run
    Authorization: Bearer <ZETARIS_API_KEY>, X-Org-ID, X-Request-ID: fresh uuid4
    {"queryId": uuid4, "sql": "SELECT ...", "source": "SqlEditor", "limit": 100}
    -> {"data": [[...], ...]}   column names may come back null, so callers rely on SELECT order

One statement per call; 5xx is retried. Never port 8889.
Env: ZETARIS_REST_URL (default http://localhost:8888), ZETARIS_API_KEY, ZETARIS_ORG_ID (default 1).
"""
import os
import time
import uuid

import requests

PATH = "/api/v1.0/sql-editor/sqls/run"


def run_sql(sql, limit=100, retries=3, backoff=1.0):
    """Run one Spark SQL statement. Returns (rows as lists, query_id)."""
    base = os.environ.get("ZETARIS_REST_URL", "http://localhost:8888").rstrip("/")
    if ":8889" in base:
        raise ValueError("ZETARIS_REST_URL must not use port 8889")
    if ";" in sql.strip().rstrip(";"):
        raise ValueError("Zetaris takes one statement per call")
    query_id = str(uuid.uuid4())
    body = {"queryId": query_id, "sql": sql, "source": "SqlEditor", "limit": limit}
    for attempt in range(retries):
        headers = {"Authorization": f"Bearer {os.environ.get('ZETARIS_API_KEY', '')}",
                   "X-Org-ID": os.environ.get("ZETARIS_ORG_ID", "1"),
                   "X-Request-ID": str(uuid.uuid4()), "Content-Type": "application/json"}
        r = requests.post(base + PATH, headers=headers, json=body, timeout=60)
        if r.status_code >= 500 and attempt < retries - 1:
            time.sleep(backoff * 2 ** attempt)
            continue
        r.raise_for_status()
        return (r.json() or {}).get("data") or [], query_id
