"""Meterless H-MEM, implemented for the Latent Raw agent (spec: github.com/meterless/H-MEM, AGENTS.md).

What is implemented, by spec section:
  §2  canonical memory record: id, content, type, layer, timestamps, access count, tags, embedding,
      confidence, source, domain, namespace, entities, relatedTo, supersedes, derivedFrom
  §3  tiered storage: short_term (this process), working (memory/hmem/working.json),
      long_term (memory/hmem/long_term.json); load = hydrate, merge, de-duplicate, backfill embeddings,
      forgetting curve on non-long-term records
  §4  mining with the no-model fallback (§4.3): direct capture of colourist rulings and run outcomes
  §5  enrichment: domain, namespace, entities (the agent's item names), tags
  §6  hybrid retrieval (vector + keyword + tag + domain + entity + layer + recency - superseded penalty,
      threshold 0.35, top 5) and reinjection: domain-grouped block plus trace metadata
  §9  trust layer: append-only audit ledger (memory/hmem/ledger.jsonl) and feedback adaptation
  §10 conflict detection (opposing phrase pairs on shared entities) and audited resolution
Not implemented (opt-in in the spec): §7 dreaming and §8 sleep cycles.
"""
import hashlib
import json
import math
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

DIM = 128
TOP_N = 5
THRESHOLD = 0.35
LAYER_WEIGHT = {"long_term": 1.0, "working": 0.8, "short_term": 0.6}
HALF_LIFE_DAYS = 14.0  # forgetting curve for working / short-term records
OPPOSING = [("may clip", "must not clip"), ("never pull", "pull down"), ("always", "never"),
            ("keep", "remove"), ("enable", "disable"), ("allow", "forbid")]
STOP = {"on", "this", "the", "a", "an", "it", "them", "of", "in", "to", "and", "or", "be", "is", "show"}


def _words(text):
    return [w for w in re.findall(r"[a-z0-9_]+", text.lower()) if w not in STOP]


def _stem(w):
    return w[:-1] if w.endswith("s") and len(w) > 3 else w


def embed(text):
    """Lightweight hashed bag-of-words embedding (§2 'lightweight embedding'); no model needed."""
    v = [0.0] * DIM
    for w in _words(text):
        for part in [_stem(w)] + _stem(w).split("_"):
            h = int(hashlib.md5(part.encode()).hexdigest(), 16)
            v[h % DIM] += 1.0 if (h >> 8) % 2 else -1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [round(x / n, 5) for x in v]


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b))


@dataclass
class Memory:
    content: str
    type: str = "preference"
    layer: str = "long_term"
    id: str = field(default_factory=lambda: "mem_" + uuid.uuid4().hex[:10])
    timestamp: float = field(default_factory=time.time)
    lastAccessed: float = field(default_factory=time.time)
    accessCount: int = 0
    tags: list = field(default_factory=list)
    embedding: list = field(default_factory=list)
    confidence: float = 0.8
    source: str = "direct"
    domain: str = "grading"
    namespace: str = "latent-raw"
    entities: list = field(default_factory=list)
    relatedTo: list = field(default_factory=list)
    supersedes: str = None
    derivedFrom: list = field(default_factory=list)
    ruling_id: str = None


class HMem:
    def __init__(self, root="memory/hmem", actor="agent"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.actor = actor
        self.short_term = []
        self.records = {}
        self.last_trace = None
        self._load()

    # ---- §3 storage ---------------------------------------------------------------------------
    def _tier_path(self, layer):
        return self.root / f"{layer}.json"

    def _load(self):
        now = time.time()
        for layer in ("long_term", "working"):
            p = self._tier_path(layer)
            for d in (json.loads(p.read_text()) if p.exists() else []):
                m = Memory(**d)
                if not m.embedding:
                    m.embedding = embed(m.content)  # backfill
                if m.layer != "long_term":  # forgetting curve
                    age_days = (now - m.lastAccessed) / 86400
                    m.confidence = round(m.confidence * 0.5 ** (age_days / HALF_LIFE_DAYS), 4)
                self.records[m.id] = m  # de-duplicate by id

    def save(self):
        for layer in ("long_term", "working"):
            rows = [asdict(m) for m in self.records.values() if m.layer == layer]
            rows.sort(key=lambda d: d["timestamp"])
            self._tier_path(layer).write_text(json.dumps(rows, indent=1) + "\n")

    # ---- §9 trust ledger --------------------------------------------------------------------------
    def audit(self, memory_id, action, previous=None, new=None, **details):
        entry = {"memoryId": memory_id, "action": action, "actor": self.actor, "timestamp": round(time.time(), 3),
                 "previousState": previous, "newState": new, "details": details or None}
        with open(self.root / "ledger.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    def ledger(self, memory_id=None):
        p = self.root / "ledger.jsonl"
        rows = [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
        return [r for r in rows if memory_id is None or r["memoryId"] == memory_id]

    # ---- §4/§5 mining + enrichment (no-model fallback) ---------------------------------------
    def capture(self, content, *, type="preference", layer="long_term", source="direct", entities=(),
                tags=(), confidence=0.8, ruling_id=None, supersedes=None, derived_from=(), domain="grading",
                namespace="latent-raw"):
        for m in self.records.values():  # idempotent capture of the same ruling
            if ruling_id and m.ruling_id == ruling_id:
                return m
        m = Memory(content=content, type=type, layer=layer, source=source, entities=sorted(set(entities))[:10],
                   tags=sorted(set(tags)), confidence=confidence, ruling_id=ruling_id, supersedes=supersedes,
                   domain=domain, namespace=namespace,
                   derivedFrom=list(derived_from), embedding=embed(content))
        if layer == "short_term":
            self.short_term.append(m)
        self.records[m.id] = m
        self.audit(m.id, "create", new=asdict(m) | {"embedding": None})
        self.scan_conflicts(m)
        return m

    # ---- §6 retrieval + reinjection -------------------------------------------------------------
    def score(self, m, q_emb, q_words, q_entities, q_domain, q_tags, now):
        superseded = any(o.supersedes == m.id for o in self.records.values())
        kw = len({_stem(w) for w in _words(m.content)} & {_stem(w) for w in q_words}) / max(1, len(set(q_words)))
        ent = len(set(m.entities) & set(q_entities)) / max(1, min(len(set(m.entities)), len(set(q_entities))))
        tag = len(set(m.tags) & set(q_tags)) / max(1, len(set(q_tags)))
        recency = 0.5 ** (((now - m.lastAccessed) / 86400) / 30.0)
        s = (0.30 * max(0.0, _cos(m.embedding, q_emb)) + 0.20 * kw + 0.20 * ent + 0.05 * tag
             + (0.10 if m.domain == q_domain else 0.0) + 0.10 * LAYER_WEIGHT[m.layer] + 0.05 * recency)
        s *= (0.5 + 0.5 * m.confidence)
        if superseded:
            s *= 0.3
        return round(s, 4)

    def retrieve(self, query, entities=(), domain="grading", tags=(), top_n=TOP_N, threshold=THRESHOLD):
        now = time.time()
        q_emb, q_words = embed(query + " " + " ".join(entities)), _words(query + " " + " ".join(entities))
        scored = sorted(((self.score(m, q_emb, q_words, list(entities), domain, list(tags), now), m)
                         for m in self.records.values()), key=lambda t: -t[0])
        hits = [(s, m) for s, m in scored if s >= threshold][:top_n]
        for s, m in hits:
            m.accessCount += 1
            m.lastAccessed = now
            self.audit(m.id, "read", details_score=s, query=query[:200])
        self.last_trace = {"retrievalReason": f"query={query[:120]!r} entities={list(entities)}",
                           "strategy": "minimal" if len(hits) <= 2 else "comprehensive",
                           "scores": {m.id: s for s, m in hits},
                           "rejected": {m.id: s for s, m in scored if s < threshold}}
        return hits

    def reinject(self, hits):
        """§6.2: domain-grouped context block for the prompt, plus trace metadata."""
        groups = {}
        for s, m in hits:
            groups.setdefault(m.domain, []).append(f"- [{m.layer}] {m.content}"
                                                    + (f" (entities: {', '.join(m.entities)})" if m.entities else ""))
        block = "\n".join(f"## {d}\n" + "\n".join(lines) for d, lines in groups.items())
        return block, self.last_trace

    # ---- §9.2 feedback --------------------------------------------------------------------------
    def feedback(self, memory_id, kind, **details):
        m = self.records[memory_id]
        before = m.confidence
        if kind == "helpful":
            m.confidence = min(1.0, m.confidence + 0.05)
            m.accessCount += 1
        elif kind == "not_helpful":
            m.confidence = max(0.0, m.confidence - 0.05)
        elif kind == "wrong":
            m.confidence = max(0.0, m.confidence - 0.2)
            m.tags = sorted(set(m.tags) | {"review"})
        else:
            raise ValueError(kind)
        self.audit(memory_id, "feedback", previous={"confidence": before}, new={"confidence": m.confidence},
                   kind=kind, **details)

    # ---- §10 conflicts ----------------------------------------------------------------------------
    def scan_conflicts(self, new):
        found = []
        text_n = new.content.lower()
        for m in self.records.values():
            if m.id == new.id or new.supersedes == m.id or m.supersedes == new.id or m.id in new.relatedTo:
                continue
            shared = set(m.entities) & set(new.entities)
            if not shared:
                continue
            text_m = m.content.lower()
            for a, b in OPPOSING:
                if (a in text_n and b in text_m) or (b in text_n and a in text_m):
                    conf = 0.6 + (0.2 if m.domain == new.domain else 0.0) + 0.1 * min(2, len(shared))
                    rec = {"a": m.id, "b": new.id, "reason": f"opposing '{a}' / '{b}' on {sorted(shared)}",
                           "confidence": round(min(conf, 1.0), 2), "resolved": None}
                    found.append(rec)
                    self.audit(new.id, "conflict_detected", other=m.id, reason=rec["reason"])
                    break
        if found:
            p = self.root / "conflicts.json"
            prior = json.loads(p.read_text()) if p.exists() else []
            p.write_text(json.dumps(prior + found, indent=1) + "\n")
        return found

    def resolve_conflicts(self):
        """Auto-decision by recency and confidence (§10.2): the newer, more confident record supersedes."""
        p = self.root / "conflicts.json"
        rows = json.loads(p.read_text()) if p.exists() else []
        for c in rows:
            if c["resolved"]:
                continue
            a, b = self.records.get(c["a"]), self.records.get(c["b"])
            if not a or not b:
                continue
            keep, drop = (b, a) if (b.timestamp, b.confidence) >= (a.timestamp, a.confidence) else (a, b)
            keep.supersedes = drop.id
            keep.relatedTo = sorted(set(keep.relatedTo) | {drop.id})[:5]
            c["resolved"] = {"strategy": "keep_" + ("b" if keep is b else "a"), "kept": keep.id, "superseded": drop.id}
            self.audit(keep.id, "conflict_resolved", other=drop.id, strategy=c["resolved"]["strategy"])
        p.write_text(json.dumps(rows, indent=1) + "\n") if rows else None
        return rows
