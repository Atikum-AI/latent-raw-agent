"""Reads through Zetaris: the client contract, and a whole case read via a fake Zetaris."""
import json

import pytest

from agent import core, demo, reason, samples, zetaris
from agent.evidence import Evidence
from evals.score import load_cases, score_case


class R:
    def __init__(self, status, body=None):
        self.status_code, self.body = status, body or {}
    def json(self): return self.body
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def test_client_contract_and_retry(monkeypatch):
    sent = []
    replies = iter([R(502), R(200, {"data": [["sky", "12.5"]]})])

    def post(url, headers, json, timeout):
        sent.append((url, headers, json))
        return next(replies)

    monkeypatch.setattr(zetaris.requests, "post", post)
    monkeypatch.setenv("ZETARIS_REST_URL", "http://localhost:8888")
    monkeypatch.setenv("ZETARIS_API_KEY", "k")
    monkeypatch.setenv("ZETARIS_ORG_ID", "1")
    rows, qid = zetaris.run_sql("SELECT item, clipped_pct FROM evidence_pg.item_metrics", backoff=0)
    assert rows == [["sky", "12.5"]]
    assert len(sent) == 2  # the 502 was retried
    url, h, body = sent[-1]
    assert url == "http://localhost:8888/api/v1.0/sql-editor/sqls/run"
    assert h["Authorization"] == "Bearer k" and h["X-Org-ID"] == "1" and h["X-Request-ID"]
    assert sent[0][1]["X-Request-ID"] != h["X-Request-ID"]
    assert body == {"queryId": qid, "sql": "SELECT item, clipped_pct FROM evidence_pg.item_metrics",
                    "source": "SqlEditor", "limit": 100}


def test_client_refuses_8889_and_two_statements(monkeypatch):
    monkeypatch.setenv("ZETARIS_REST_URL", "http://localhost:8889")
    with pytest.raises(ValueError, match="8889"):
        zetaris.run_sql("SELECT 1")
    monkeypatch.setenv("ZETARIS_REST_URL", "http://localhost:8888")
    with pytest.raises(ValueError, match="one statement"):
        zetaris.run_sql("SELECT 1; DROP TABLE shots")


def test_spark_sql_has_prefix_and_values():
    ev = Evidence(url="")
    sql = ev.to_spark("SELECT evidence_id, item FROM item_metrics WHERE shot_id IN (?, ?) AND frame = ?",
                      ["clip03a_canoe_dawn", "clip03b_jungle_warm@balance", 48])
    assert sql == ("SELECT evidence_id, item FROM evidence_pg.item_metrics "
                   "WHERE shot_id IN ('clip03a_canoe_dawn', 'clip03b_jungle_warm@balance') AND frame = 48")
    with pytest.raises(ValueError):
        ev.to_spark("SELECT item FROM item_metrics WHERE item = ?", ["x' OR '1'='1"])


def test_zetaris_without_postgres_is_refused(monkeypatch):
    with pytest.raises(RuntimeError, match="EVIDENCE_DATABASE_URL"):
        Evidence(url="", read_via="zetaris")
    assert Evidence(url="", read_via="zetaris", sample_mode=True).read_via == "local"


def test_whole_case_reads_through_zetaris(tmp_path, monkeypatch):
    """Writes land in the database; every read goes out as Spark SQL and comes back as
    nameless lists of text, like Zetaris. The run must score the same as a direct read."""
    monkeypatch.setattr(samples, "SAMPLES_DIR", demo.make(tmp_path / "samples"))
    monkeypatch.setattr(reason, "CACHE", tmp_path / "cache.json")
    monkeypatch.setattr(reason.nightshift, "chat", lambda *a, **k: (_ for _ in ()).throw(ConnectionError()))
    ev = Evidence(url="")
    ev.read_via = "zetaris"  # the test database stands in for Postgres behind Zetaris
    calls = []

    def fake_run_sql(sql, **k):
        calls.append(sql)
        cur = ev.conn.execute(sql.replace("evidence_pg.", ""))
        return [[None if v is None else str(v) for v in row] for row in cur.fetchall()], "q-1"

    monkeypatch.setattr(zetaris, "run_sql", fake_run_sql)
    case = json.load(open("evals/cases/case03_dawn_jungle_match.json"))
    out = core.Run(case["case_id"], case["clips"], log_path=tmp_path / "r.jsonl", evidence=ev).go()
    trace = [json.loads(l) for l in (tmp_path / "r.jsonl").read_text().splitlines()]
    assert out["decision"]["passed"] is True
    assert score_case(load_cases("evals/cases")[case["case_id"]], trace)["score"] == 100.0
    assert calls and all("evidence_pg." in c and "?" not in c for c in calls)
    sql = [q for l in trace for q in l["sql"]]
    assert sql and all(q["via"] == "zetaris" and q["query_id"] == "q-1" for q in sql)
