"""H-MEM conformance checks for the spec sections we implement (github.com/meterless/H-MEM, AGENTS.md)."""
import json
import time

from agent.hmem import THRESHOLD, HMem

FLAME = "On this show, practical flames may clip. Never pull them down."


def test_capture_has_the_canonical_record_and_is_audited(tmp_path):  # §2, §9.1
    mem = HMem(tmp_path)
    m = mem.capture(FLAME, entities=["flame", "practical"], tags=["ruling"], ruling_id="practical_flames_hot")
    for f in ("id", "content", "type", "layer", "timestamp", "lastAccessed", "accessCount", "tags", "embedding",
              "confidence", "source", "domain", "namespace", "entities", "relatedTo", "supersedes", "derivedFrom"):
        assert hasattr(m, f)
    assert [e["action"] for e in mem.ledger(m.id)] == ["create"]


def test_tiers_persist_and_reload(tmp_path):  # §3
    mem = HMem(tmp_path)
    lt = mem.capture(FLAME, layer="long_term", entities=["flame"])
    wk = mem.capture("Sky may sit at the ceiling on dawn shots.", layer="working", entities=["sky"])
    st = mem.capture("This session: client prefers warm skin.", layer="short_term", entities=["skin"])
    mem.save()
    again = HMem(tmp_path)
    assert lt.id in again.records and wk.id in again.records and st.id not in again.records


def test_hybrid_retrieval_finds_the_ruling_by_item_and_skips_noise(tmp_path):  # §6.1
    mem = HMem(tmp_path)
    flame = mem.capture(FLAME, entities=["flame", "practical"], tags=["ruling"], ruling_id="practical_flames_hot")
    mem.capture("Invoices go out on Fridays.", domain="admin", entities=["invoice"])
    hits = mem.retrieve("colourist ruling for girl hands lantern_flame table",
                        entities=["flame", "girl", "hand", "lantern", "table"], tags=["ruling"])
    assert [m.id for _, m in hits] == [flame.id]
    assert hits[0][0] >= THRESHOLD
    assert mem.last_trace["retrievalReason"] and flame.id in mem.last_trace["scores"]


def test_reinjection_is_domain_grouped_with_layer_markers(tmp_path):  # §6.2
    mem = HMem(tmp_path)
    mem.capture(FLAME, entities=["flame"], tags=["ruling"])
    block, trace = mem.reinject(mem.retrieve("flame ruling", entities=["flame"], tags=["ruling"]))
    assert block.startswith("## grading") and "[long_term]" in block and "strategy" in trace


def test_feedback_moves_confidence_and_is_audited(tmp_path):  # §9.2
    mem = HMem(tmp_path)
    m = mem.capture(FLAME, entities=["flame"], confidence=0.8)
    mem.feedback(m.id, "helpful")
    assert m.confidence > 0.8
    mem.feedback(m.id, "wrong")
    assert m.confidence < 0.8 and "review" in m.tags
    assert [e["action"] for e in mem.ledger(m.id)] == ["create", "feedback", "feedback"]


def test_conflict_is_detected_and_resolved_by_recency(tmp_path):  # §10
    mem = HMem(tmp_path)
    old = mem.capture("Practical flames must not clip: pull down the lantern.", entities=["flame", "practical"])
    old.timestamp -= 86400
    new = mem.capture(FLAME, entities=["flame", "practical"])
    conflicts = json.loads((tmp_path / "conflicts.json").read_text())
    assert conflicts and conflicts[0]["a"] == old.id and conflicts[0]["b"] == new.id
    mem.resolve_conflicts()
    assert new.supersedes == old.id
    hits = dict((m.id, s) for s, m in mem.retrieve("flame ruling", entities=["flame", "practical"], threshold=0))
    assert hits[new.id] > hits[old.id]  # superseded penalty
    assert "conflict_resolved" in [e["action"] for e in mem.ledger(new.id)]


def test_forgetting_curve_only_touches_non_long_term(tmp_path):  # §3 step 5
    mem = HMem(tmp_path)
    lt = mem.capture(FLAME, layer="long_term", confidence=0.8)
    wk = mem.capture("Dawn skies may clip.", layer="working", confidence=0.8)
    wk.lastAccessed -= 28 * 86400
    lt.lastAccessed -= 28 * 86400
    mem.save()
    again = HMem(tmp_path)
    assert again.records[lt.id].confidence == 0.8 and again.records[wk.id].confidence < 0.3
