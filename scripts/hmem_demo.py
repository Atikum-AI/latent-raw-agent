"""H-MEM in one minute: the colourist changes her mind, the memory notices, and the agent follows.

    python scripts/hmem_demo.py            # uses a throwaway folder, prints every step
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.hmem import HMem  # noqa: E402

root = Path(tempfile.mkdtemp(prefix="hmem_demo_"))
mem = HMem(root, actor="colourist")

print("1. Last week the colourist ruled: lanterns must not clip.")
old = mem.capture("Practical flames must not clip: pull down the lantern.", entities=["flame", "practical"],
                  tags=["ruling"], ruling_id="flames_held_down", source="session:2026-10-20")
old.timestamp -= 7 * 86400

print("2. Today she ruled the opposite: practical flames may clip, never pull them down.")
new = mem.capture("On this show, practical flames may clip. Never pull them down.", entities=["flame", "practical"],
                  tags=["ruling"], ruling_id="practical_flames_hot", source="session:2026-10-27")
for c in json.loads((root / "conflicts.json").read_text()):
    print(f"   conflict detected: {c['reason']} (confidence {c['confidence']})")

print("3. Resolution by recency and confidence (audited):")
for c in mem.resolve_conflicts():
    print(f"   kept {c['resolved']['kept']}, superseded {c['resolved']['superseded']}")

print("4. The agent measures a lantern shot and asks H-MEM which rulings apply:")
mem.actor = "agent"
hits = mem.retrieve("colourist ruling for girl hands lantern_flame table",
                    entities=["flame", "girl", "hand", "lantern", "table"], tags=["ruling"], threshold=0)
for s, m in hits:
    print(f"   {s:.3f}  {m.ruling_id:20s} {'(superseded)' if any(o.supersedes == m.id for o in mem.records.values()) else ''}")
block, trace = mem.reinject(hits[:1])
print("   reinjected into the planner's prompt:\n   " + block.replace("\n", "\n   "))

print("5. The balance pass kept the flame hot and verify passed, so the ruling was helpful:")
mem.feedback(new.id, "helpful", verify_passed=True)
print(f"   confidence now {mem.records[new.id].confidence:.2f}")

print("6. Audit ledger (append-only):")
for e in mem.ledger():
    print(f"   {e['actor']:10s} {e['action']:18s} {e['memoryId']}")
